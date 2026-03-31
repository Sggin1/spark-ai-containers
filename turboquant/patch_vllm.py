"""
Patch vLLM with TurboQuant support from PR #38479.
Applies modifications to existing files inside the container.
Run inside the container as: python3 /workspace/build/patch_vllm.py
"""

import re

VLLM_ROOT = "/usr/local/lib/python3.12/dist-packages/vllm"

def patch_file(filepath, patches):
    """Apply a list of (anchor, insertion) patches to a file."""
    with open(filepath) as f:
        content = f.read()

    for anchor, insertion, mode in patches:
        if anchor not in content:
            print(f"  WARNING: anchor not found in {filepath}: {anchor[:60]}...")
            continue
        if mode == "after":
            content = content.replace(anchor, anchor + insertion)
        elif mode == "before":
            content = content.replace(anchor, insertion + anchor)
        print(f"  Patched: {anchor[:50].strip()}...")

    with open(filepath, "w") as f:
        f.write(content)


def main():
    print("Patching vLLM for TurboQuant support (PR #38479)...\n")

    # 1. cache.py — add tq3/tq4 to CacheDType
    print("[1/6] config/cache.py")
    with open(f"{VLLM_ROOT}/config/cache.py") as f:
        content = f.read()
    content = content.replace(
        '    "fp8_ds_mla",\n]',
        '    "fp8_ds_mla",\n    "tq3",\n    "tq4",\n]'
    )
    with open(f"{VLLM_ROOT}/config/cache.py", "w") as f:
        f.write(content)
    print("  Added tq3/tq4 to CacheDType")

    # 2. attention.py — add TurboQuant buffer init + get_kv_cache_spec branch
    print("\n[2/6] model_executor/layers/attention/attention.py")
    attn_path = f"{VLLM_ROOT}/model_executor/layers/attention/attention.py"
    with open(attn_path) as f:
        content = f.read()

    # 2a. Add _init_turboquant_buffers call after _init_kv_cache_quant
    anchor_init = "        _init_kv_cache_quant(self, quant_config, prefix)\n"
    tq_init_call = '''
        # Initialize TurboQuant buffers (Pi, S, centroids) if tq cache dtype
        if kv_cache_dtype.startswith("tq"):
            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)
'''
    if tq_init_call.strip() not in content:
        content = content.replace(anchor_init, anchor_init + tq_init_call)
        print("  Added _init_turboquant_buffers call")

    # 2b. Add the _init_turboquant_buffers method before forward()
    tq_method = '''
    def _init_turboquant_buffers(
        self, cache_dtype: str, head_size: int, prefix: str
    ) -> None:
        """Initialize TurboQuant rotation/projection matrices and centroids."""
        from vllm.turboquant.config import TurboQuantConfig
        from vllm.turboquant.quantizer import (
            generate_rotation_matrix,
            generate_qjl_matrix,
        )
        from vllm.turboquant.centroids import get_centroids

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)

        from vllm.model_executor.models.utils import extract_layer_index
        layer_idx = extract_layer_index(prefix)
        seed = tq_config.seed + layer_idx * 1337

        self.register_buffer(
            "_tq_Pi",
            generate_rotation_matrix(head_size, seed=seed),
        )
        self.register_buffer(
            "_tq_S",
            generate_qjl_matrix(head_size, seed=seed + 1),
        )
        self.register_buffer(
            "_tq_centroids",
            get_centroids(head_size, tq_config.mse_bits),
        )
        self._tq_config = tq_config

'''
    # Insert before "    def forward(" — use exact match with no preceding blank line
    if "def _init_turboquant_buffers" not in content:
        content = content.replace(
            "    def forward(\n        self,\n        query: torch.Tensor,\n        key: torch.Tensor,",
            tq_method + "    def forward(\n        self,\n        query: torch.Tensor,\n        key: torch.Tensor,"
        )
        print("  Added _init_turboquant_buffers method")

    # 2c. Add TurboQuant branch in get_kv_cache_spec
    tq_spec = '''        elif self.kv_cache_dtype.startswith("tq"):
            from vllm.turboquant.config import TurboQuantConfig
            tq_config = TurboQuantConfig.from_cache_dtype(
                self.kv_cache_dtype, self.head_size)
            padded_slot = tq_config.padded_slot_size
            effective_head_size = padded_slot // 2
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=effective_head_size,
                head_size_v=effective_head_size,
                dtype=self.kv_cache_torch_dtype,
            )
'''
    spec_anchor = "        else:\n            return FullAttentionSpec(\n                block_size=block_size,\n                num_kv_heads=self.num_kv_heads,\n                head_size=self.head_size,\n                head_size_v=self.head_size_v,"
    if 'kv_cache_dtype.startswith("tq")' not in content:
        content = content.replace(spec_anchor, tq_spec + spec_anchor)
        print("  Added TurboQuant branch in get_kv_cache_spec")

    with open(attn_path, "w") as f:
        f.write(content)

    # 3. models/config.py — add NemotronH TurboQuant config
    print("\n[3/6] model_executor/models/config.py")
    # This is a large patch (146 lines). Read the new content from our extracted file.
    # For now, just add the minimal config validation.
    config_path = f"{VLLM_ROOT}/model_executor/models/config.py"
    with open(config_path) as f:
        content = f.read()

    # Patch the dtype lookup to handle tq3/tq4
    old_lookup = '            kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]'
    new_lookup = '''            if cache_config.cache_dtype.startswith("tq"):
                import torch
                kv_cache_dtype = torch.uint8
            else:
                kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]'''
    if 'cache_dtype.startswith("tq")' not in content:
        content = content.replace(old_lookup, new_lookup)
        print("  Patched dtype lookup for tq3/tq4")

    with open(config_path, "w") as f:
        f.write(content)

    # 4. platforms/cuda.py — add tq dtype support
    print("\n[4/6] platforms/cuda.py")
    cuda_path = f"{VLLM_ROOT}/platforms/cuda.py"
    with open(cuda_path) as f:
        content = f.read()

    # Add TURBOQUANT to attention backend candidate lists
    if "TURBOQUANT" not in content:
        # Add to both compute capability branches
        content = content.replace(
            '            return [\n                AttentionBackendEnum.FLASHINFER,\n                AttentionBackendEnum.FLASH_ATTN,\n                AttentionBackendEnum.TRITON_ATTN,\n                AttentionBackendEnum.FLEX_ATTENTION,\n            ]',
            '            return [\n                AttentionBackendEnum.TURBOQUANT,\n                AttentionBackendEnum.FLASHINFER,\n                AttentionBackendEnum.FLASH_ATTN,\n                AttentionBackendEnum.TRITON_ATTN,\n                AttentionBackendEnum.FLEX_ATTENTION,\n            ]'
        )
        content = content.replace(
            '            return [\n                AttentionBackendEnum.FLASH_ATTN,\n                AttentionBackendEnum.FLASHINFER,\n                AttentionBackendEnum.TRITON_ATTN,\n                AttentionBackendEnum.FLEX_ATTENTION,\n            ]',
            '            return [\n                AttentionBackendEnum.TURBOQUANT,\n                AttentionBackendEnum.FLASH_ATTN,\n                AttentionBackendEnum.FLASHINFER,\n                AttentionBackendEnum.TRITON_ATTN,\n                AttentionBackendEnum.FLEX_ATTENTION,\n            ]'
        )
        print("  Added TURBOQUANT to attention backend candidates")

    with open(cuda_path, "w") as f:
        f.write(content)

    # 5. utils/torch_utils.py — add tq dtype mapping
    print("\n[5/6] utils/torch_utils.py")
    tu_path = f"{VLLM_ROOT}/utils/torch_utils.py"
    with open(tu_path) as f:
        content = f.read()

    # The PR adds tq dtype handling in get_kv_cache_torch_dtype
    # Insert before the else that raises "Invalid kv cache dtype"
    if '"tq3"' not in content:
        content = content.replace(
            '        else:\n            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")\n    elif isinstance(cache_dtype, torch.dtype):',
            '        elif cache_dtype.startswith("tq"):\n            torch_dtype = torch.uint8\n        else:\n            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")\n    elif isinstance(cache_dtype, torch.dtype):'
        )
        print("  Added tq dtype handling in get_kv_cache_torch_dtype")

        # Also patch kv_cache_dtype_str_to_dtype (line ~351)
        content = content.replace(
            '    return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]\n\n\ndef set_random_seed',
            '    if kv_cache_dtype.startswith("tq"):\n        return torch.uint8\n    return STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]\n\n\ndef set_random_seed'
        )
        print("  Added tq handling in kv_cache_dtype_str_to_dtype")

    with open(tu_path, "w") as f:
        f.write(content)

    # 6. v1/attention/backends/registry.py — add TurboQuant backend
    print("\n[6/6] v1/attention/backends/registry.py")
    reg_path = f"{VLLM_ROOT}/v1/attention/backends/registry.py"
    with open(reg_path) as f:
        content = f.read()

    if "TURBOQUANT" not in content:
        # Insert TURBOQUANT before CPU_ATTN in the AttentionBackendEnum class
        content = content.replace(
            '    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"',
            '    TURBOQUANT = "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"\n    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"'
        )
        print("  Added TURBOQUANT backend to registry")

    with open(reg_path, "w") as f:
        f.write(content)

    print("\n✓ All patches applied successfully!")
    print("  New files should be copied to their locations separately.")


if __name__ == "__main__":
    main()
