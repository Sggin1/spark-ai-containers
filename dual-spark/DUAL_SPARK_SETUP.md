# DGX Spark Pair — Setup & Troubleshooting (Single Cable, 200 Gb/s)

One canonical guide for pairing two NVIDIA DGX Sparks (GB10, SM121) with a
single CX-7 direct-link cable, running vLLM via eugr's
[`spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker). Every step is
paste-ready. Troubleshooting lives next to the step that introduced it.

**Target result:** TP=2 cluster over ~195 Gb/s RDMA (~98 % of the single-port
200 Gb/s ceiling), inference healthy in under 20 minutes on a pair that
already boots.

If you're scaling to 3+ Sparks you'll want the mesh / 4-port topology —
out of scope here; see
<https://forums.developer.nvidia.com/t/6x-spark-setup/354399/56>.

---

## Zero — Power (check this before chasing network bugs)

DGX Spark can spike past **300 W** on inference transients. If the AC path
sags (undersized UPS, overloaded circuit, weak strip), the governor can
clamp the GPU into **P8 while under load**. Benchmarks then plateau at
~1/3 of spec with no clear log line — we burned hours on that.

**This is not “all UPS bad.”** Many people run fine on UPS. What we
verified is one failure mode on one path:

| AC source (same box, same code) | Burst / sustained (bf16 GEMM) |
|---|---|
| One consumer UPS under transient load | 8.9 / 46.9 TFLOP/s |
| Direct wall outlet | 82 / 120 TFLOP/s |

**Oracle — run before any networking work:**

```bash
# Healthy GB10 ≈ 80–125 burst, 80–140 sustained. < ~60 either number → fix power path first.
docker run --rm --gpus all --ipc=host \
  -v $PWD/gpu_stress.py:/work/gpu_stress.py \
  vllm-node-tf5 python3 -u /work/gpu_stress.py
```

If numbers are healthy on your UPS (or PDU / strip), keep it. If not: try
wall (or a higher-capacity source), re-run, then resume setup.

Ignore `dmesg | grep "insufficient power"` — on Spark those mlx5 PCIe
warnings fire at boot even on healthy power (chassis quirk). Trust the
TFLOP/s number, not that line.

---

## TL;DR — seven things that go wrong (single cable)

| # | Symptom | Fix |
|---|---|---|
| 0 | Inference plateaus at ~1/3 spec regardless of config | Power sag / P8 under load — verify AC with `gpu_stress.py` (§0) |
| 1 | Launcher SSHs as wrong user → `Permission denied (publickey)` | Prefer one shared user (§1.1 A); or two-user `~/.ssh/config` map (§2.4) |
| 2 | Ray never connects, `nc -zv 10.10.0.1 29501` hangs | UFW open 10.10.0.0/24 + 10.10.1.0/24 on both (§2.2) |
| 3 | Only ~100 Gb/s total; half the fabric idle | Twins on **different** /24 subnets (§2.1) |
| 4 | `ibv_modify_qp failed 101 Network is unreachable` | `NCCL_IB_GID_INDEX=3` (RoCEv2 + IPv4-mapped) (§4.4) |
| 5 | NCCL falls back to TCP over mgmt | Set `NCCL_NET_PLUGIN=none`, `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_MERGE_NICS=0` (§4.5) |
| 6 | `rsync` / `scp` tops out ~150 MB/s | Use local docker registry on `10.10.0.1:5000` (§2.5) |
| 7 | vLLM cluster won't start: "current node has no GPU available" | Stale Ray placement group — `launch-cluster.sh stop` + `sudo pkill -f ray::` on both |

---

## 1. Two decisions to make before anything else

### 1.1 Username model: one user (A, recommended) or two (B, works)

Launchers (eugr's `spark-vllm-docker`, sparkrun, NVIDIA playbooks) SSH to
the peer **as the current username**. Two workable models:

| Model | What it looks like | Headwind | Playbook |
|---|---|---|---|
| **A · same user on both** (recommended) | both nodes log in as `<user>` (same UID) | Lowest | §1.1.1 + §2.3 A |
| **B · different user per box** | `<user_a>` head, `<user_b>` worker | SSH map on every fabric IP | §1.1.2 + §2.3 B + §2.4 |
| **B → A** (convert later) | were two users; now one shared | One-time migration | §1.1.3 |

**We ran B first, then converted to A.** B is fine while experimenting; A
is less day-to-day friction. Prefer A at install if you can. Both are
documented end-to-end below — there is no “unsupported” path, only more
or less headwind.

File ownership travels by **UID**, not name — matching UIDs keeps rsynced
trees and shared paths sane even before names match.

#### 1.1.1 Playbook A — same user from day one

```bash
# pick one name + one UID; run on BOTH nodes
sudo useradd -u 1001 -m -G sudo,docker <user>
# set password / SSH keys as you normally would, then:
```

| Step | Where | Action |
|---|---|---|
| 1 | both | same `<user>` + same UID in `sudo,docker` |
| 2 | head | §2.3 model A — `ssh-copy-id <user>@10.10.0.2` and `…@10.10.1.2` |
| 3 | — | **skip §2.4** entirely |
| 4 | both | `/mnt/ai` symlink (§2.6) — still recommended |
| 5 | head | `ssh 10.10.0.2 whoami` → prints `<user>` (same as local) |

#### 1.1.2 Playbook B — two users (works)

Keep the stock accounts if you already created different names at install.

| Step | Where | Action |
|---|---|---|
| 1 | both | leave `<user_a>` / `<user_b>` as-is; put both in `docker` (+ `sudo` if needed) |
| 2 | head | §2.3 model B — `ssh-copy-id <user_b>@10.10.0.2` (and twin-2 IP) |
| 3 | head (+ worker if tools SSH back) | §2.4 `Host` blocks mapping fabric IPs → peer username |
| 4 | both | **do** §2.6 `/mnt/ai` — bind mounts must not depend on `$HOME` |
| 5 | head | `ssh 10.10.0.2 whoami` → prints `<user_b>`, no password |

Day-to-day cost: any new hostname/IP you SSH to needs a `Host` line, or
launchers break with `Permission denied (publickey)`.

#### 1.1.3 Playbook B → A — convert to one user

When the SSH map gets old, pick one shared name (often the head’s account
name, or a fresh `<user>`) and the same UID on both boxes.

```bash
# --- on BOTH nodes (adjust names/UID to your choice) ---
# If <user> does not exist yet on this box:
sudo useradd -u 1001 -m -G sudo,docker <user>
# If it exists but UID differs, prefer creating a clean shared UID rather
# than renumbering a busy account mid-flight.

# ensure docker group
sudo usermod -aG docker <user>

# --- data: prefer the fixed path you already use ---
# If you have /mnt/ai → real tree, keep it; chown to the shared UID:
sudo chown -R <user>:<user> /mnt/ai   # only if that tree should belong to <user>

# optional: copy old home bits you still need
# sudo rsync -aHAX /home/<user_a>/projects/ /home/<user>/projects/
```

```bash
# --- SSH as the shared user (from head, logged in as <user>) ---
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519   # skip if key exists
ssh-copy-id <user>@10.10.0.2
ssh-copy-id <user>@10.10.1.2
ssh <user>@10.10.0.2 true && echo OK
```

```bash
# --- remove model-B Host overrides on head (and worker if any) ---
# Edit ~/.ssh/config: delete User <user_b> / User <user_a> blocks for
# 10.10.0.1, 10.10.0.2, 10.10.1.1, 10.10.1.2 (or comment them out).
# After that, plain SSH must use the default local username on both sides.
```

| Verify | Expect |
|---|---|
| `whoami` on both (interactive login as shared user) | same string |
| `id -u` on both | same number |
| `ssh 10.10.0.2 whoami` from head | same `<user>`, no prompt |
| `ssh 10.10.1.2 whoami` from head | same |
| launcher / `launch-cluster.sh` SSH steps | no `Permission denied (publickey)` |

You can leave the old account on disk; just stop using it for cluster
ops. Old `Host` lines that force `User <user_b>` will fight model A —
remove them.

### 1.2 Each twin on its own /24 subnet

The single CX-7 cable is **one ~200 Gb/s fabric** split across two PCIe
slots / “twins” (~half each useful). Picture + PCI map:
[`TOPOLOGY.md`](./TOPOLOGY.md). To use both halves, each twin needs an IP
on a **different** /24.

```
        GX_NODE_1 (head)              GX_NODE_2 (worker)
twin 1:     10.10.0.1/24    ──────────    10.10.0.2/24
twin 2:     10.10.1.1/24    ──────────    10.10.1.2/24
```

Putting both twins on the same /24 (e.g. `10.10.0.3` / `10.10.0.4`) confuses
routing, NCCL subnet-aware routing silently fails, and you cap at the speed
of one twin (~100 Gb/s). eugr's `autodiscover.sh` will error explicitly if
you do this.

### Hostnames used below

Throughout the rest of this guide we'll call the two boxes `GX_NODE_1` and
`GX_NODE_2` — substitute your own hostnames freely. `GX_NODE_1` plays the
Ray **head** role (cluster master, model launcher, registry host).
`GX_NODE_2` plays the **worker** role. Where the text says "on head:" or
"on worker:" later, that refers to the role, not the hostname:

- **GX_NODE_1 (head role):** hostname `GX_NODE_1`, twins `10.10.0.1`, `10.10.1.1`
- **GX_NODE_2 (worker role):** hostname `GX_NODE_2`, twins `10.10.0.2`, `10.10.1.2`
- **User:** model A → same `<user>` on both; model B → `<user_a>` head /
  `<user_b>` worker + §2.4

---

## 2. Setup (paste-ready)

### 2.1 Assign IPs on both twins (both nodes), MTU 9000

`nmcli` ships on DGX Spark. Use the QSFP interfaces that are `Up` per
`ibdev2netdev` — names may be `enp1s0f1np1` / `enP2p1s0f1np1` or
`enp1s0f0np0` / `enP2p1s0f0np0` depending on which port you cabled.

**On head:**

```bash
sudo nmcli con add type ethernet ifname <twin1-iface> con-name cx7-twin1 \
  ipv4.addresses 10.10.0.1/24 ipv4.method manual ipv6.method disabled \
  802-3-ethernet.mtu 9000 connection.autoconnect yes
sudo nmcli con up cx7-twin1

sudo nmcli con add type ethernet ifname <twin2-iface> con-name cx7-twin2 \
  ipv4.addresses 10.10.1.1/24 ipv4.method manual ipv6.method disabled \
  802-3-ethernet.mtu 9000 connection.autoconnect yes
sudo nmcli con up cx7-twin2
```

**On worker:** same commands, `.2` addresses.

**Verify:**

```bash
ip -br addr | grep 10.10      # both twins UP with correct IPs
ping -c 2 -M do -s 8000 10.10.1.2    # second subnet + MTU 9000 both work
```

### 2.2 Open UFW for the fabric (both nodes)

DGX Spark ships UFW enabled, default DROP, allow only `192.168.0.0/24`. Ray
GCS on TCP 29501 is blocked by default.

```bash
sudo ufw allow from 10.10.0.0/24
sudo ufw allow from 10.10.1.0/24
sudo ufw reload
```

**Verify from worker:**

```bash
nc -zv 10.10.0.1 29501      # must succeed, not hang
```

### 2.3 Passwordless SSH over the fabric (head node)

**Model A** (same `<user>` both sides):

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519    # skip if key exists
ssh-copy-id <user>@10.10.0.2
ssh-copy-id <user>@10.10.1.2
ssh <user>@10.10.0.2 true && echo OK                # must be non-interactive
```

**Model B** (different user on worker): use the *worker* username on
`ssh-copy-id`, then continue to §2.4 so plain `ssh 10.10.0.2` still works
without embedding the foreign name:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ssh-copy-id <user_b>@10.10.0.2
ssh-copy-id <user_b>@10.10.1.2
```

Accept host keys for **both** fabric IPs (each is a separate SSH host).

### 2.4 Two-user model: `~/.ssh/config` map

**Skip this entire section for model A** (same username both nodes).

For model B, launchers and scripts often SSH as *your* local username to
the peer IP. Map every fabric address the head might touch to the
worker's real account — on the **head** node:

```sshconfig
Host 10.10.0.2
    HostName 10.10.0.2
    User <user_b>
    IdentityFile ~/.ssh/id_ed25519

Host 10.10.1.2
    HostName 10.10.1.2
    User <user_b>
    IdentityFile ~/.ssh/id_ed25519
```

If the worker also SSHs back to the head (some tools do), mirror the
map there with `User <user_a>` on `10.10.0.1` / `10.10.1.1`.

Verify from head:

```bash
ssh 10.10.0.2 whoami    # must print <user_b>, no password prompt
ssh 10.10.1.2 whoami    # same
```

**Still more headwind than model A:** any new script that SSHes by a
hostname you forgot to list, or rsyncs into the wrong `$HOME`, will
bite. `/mnt/ai` (§2.6) and same-UID shared data help. When you are ready
to drop the map, use the **B → A** conversion playbook in §1.1.3.

### 2.5 Local docker registry on the fabric IP (recommended)

SSH crypto is CPU-bound on GB10's Arm cores and caps around ~150 MB/s. For
bulk image and model shard transfer, run a plain `registry:2` bound to the
head node's fabric IP — transfers then run at link speed.

**On head (one-time):**

```bash
sudo mkdir -p /srv/registry
docker run -d --restart=always --name registry \
  -p 10.10.0.1:5000:5000 -v /srv/registry:/var/lib/registry \
  registry:2
```

**On both**, edit `/etc/docker/daemon.json`:

```json
{
  "runtimes": {"nvidia": {"args": [], "path": "nvidia-container-runtime"}},
  "insecure-registries": ["10.10.0.1:5000"]
}
```

Then `sudo systemctl restart docker` on both.

**Verify from worker:** `curl http://10.10.0.1:5000/v2/_catalog` →
`{"repositories":[]}`.

### 2.6 Shared AI asset path

Docker `-v` bind mounts send the host path from head to worker. If paths
differ (`$HOME/ai` resolves differently per user — common under model B),
mounts break silently. Model A still benefits from a fixed symlink.

**On both:**

```bash
mkdir -p ~/ai/{cache/hub,models,containers,datasets}
sudo ln -s "$HOME/ai" /mnt/ai
```

Add to `~/.bashrc` on both:

```bash
export HF_HOME=/mnt/ai/cache/hub
```

Reopen shell. All recipes and scripts reference `/mnt/ai` consistently
regardless of which user is running them.

### 2.7 Clone & patch the launcher

**On head:**

```bash
cd ~/projects
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Patch `autodiscover.sh` — add direct-link NCCL overrides + correct GID
detection to the 2-NIC (non-mesh) branch. Find the block that starts
`if [[ "$num_up" -eq 2 ]]; then` (around line 121) and add, before the
`elif [[ "$num_up" -eq 4 ]]`:

```bash
export DOTENV_CONTAINER_NCCL_NET_PLUGIN=none
export DOTENV_CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=1
export DOTENV_CONTAINER_NCCL_IB_MERGE_NICS=0
for hca in "${DETECTED_IB_IFS[@]}"; do
  found=""
  for i in 0 1 2 3 4 5 6 7; do
    type=$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/$i 2>/dev/null)
    gid=$(cat  /sys/class/infiniband/"$hca"/ports/1/gids/$i           2>/dev/null)
    if [[ "$type" == "RoCE v2" ]] && [[ "$gid" == 0000:0000:0000:0000:0000:ffff:* ]]; then
      found="$i"; break
    fi
  done
  [[ -n "$found" ]] && export DOTENV_CONTAINER_NCCL_IB_GID_INDEX="$found" && break
done
```

In `save_config`, change the mesh-only NCCL block to plugin-agnostic:

```bash
if [[ -n "${DOTENV_CONTAINER_NCCL_NET_PLUGIN:-}" ]]; then
  echo "CONTAINER_NCCL_NET_PLUGIN=${DOTENV_CONTAINER_NCCL_NET_PLUGIN}"
  echo "CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING=${DOTENV_CONTAINER_NCCL_IB_SUBNET_AWARE_ROUTING}"
  echo "CONTAINER_NCCL_IB_MERGE_NICS=${DOTENV_CONTAINER_NCCL_IB_MERGE_NICS}"
  [[ -n "${DOTENV_CONTAINER_NCCL_IB_GID_INDEX:-}" ]] && \
    echo "CONTAINER_NCCL_IB_GID_INDEX=${DOTENV_CONTAINER_NCCL_IB_GID_INDEX}"
fi
```

### 2.8 Run autodiscover + launch test

```bash
./autodiscover.sh        # detects interfaces, peers, writes .env
cat .env                 # expect CONTAINER_NCCL_IB_GID_INDEX=3 and 3 ring overrides
./launch-cluster.sh      # starts container on both nodes
```

---

## 3. Validation checklist

### 3.1 Both twins up with IPs (both nodes)

```bash
ip -br addr | grep 10.10
# head:   10.10.0.1/24 and 10.10.1.1/24
# worker: 10.10.0.2/24 and 10.10.1.2/24
```

### 3.2 Jumbo frames end-to-end

```bash
ping -c 2 -M do -s 8000 10.10.0.2
ping -c 2 -M do -s 8000 10.10.1.2
```

0% loss both. `-M do -s 8000` forces no fragmentation, validating MTU 9000.

### 3.3 RDMA bandwidth per twin

```bash
sudo apt-get install -y perftest

# Receiver on worker:
ssh <user>@10.10.0.2 'ib_write_bw -d rocep1s0f0 --report_gbits -q 4 -R --force-link IB &'

# Client on head:
ib_write_bw 10.10.0.2 -d rocep1s0f0 --report_gbits -q 4 -R --force-link IB
# Expect: BW average ~107 Gb/s per twin
```

Repeat with `roceP2p1s0f0` against `10.10.1.2`. Both twins together ≈
195-197 Gb/s aggregate (97-98% of 200 theoretical).

### 3.4 Ray GCS port reachable

```bash
nc -zv 10.10.0.1 29501   # from worker, must succeed not hang
```

### 3.5 Passwordless SSH over fabric

```bash
ssh 10.10.0.2 hostname
ssh 10.10.1.2 hostname
```

Both print peer hostname without prompt.

### 3.6 Container sees the right NCCL env after launch

```bash
docker logs vllm_node 2>&1 | grep -E 'NCCL_IB_GID_INDEX|NCCL_NET_PLUGIN|NCCL_IB_SUBNET_AWARE_ROUTING|NCCL_IB_MERGE_NICS'
```

Expected:

```
NCCL_IB_GID_INDEX=3
NCCL_NET_PLUGIN=none
NCCL_IB_SUBNET_AWARE_ROUTING=1
NCCL_IB_MERGE_NICS=0
```

Also verify RDMA transport (not TCP fallback):

```bash
docker logs vllm_node 2>&1 | grep -E 'NET/IB|NET/Socket'
# Want: "Initialized NET plugin IB" and "Channel ... via NET/IB/*"
# Bad:  "NET/Socket" anywhere
```

### 3.7 GPU delivering spec (post-power-fix sanity)

```bash
docker run --rm --gpus all --ipc=host \
  -v $PWD/gpu_stress.py:/work/gpu_stress.py \
  vllm-node-tf5 python3 -u /work/gpu_stress.py
# Expect 80-125 TFLOP/s burst, 80-140 sustained
```

---

## 4. Troubleshooting (problem → cause → fix)

### 4.1 Different usernames: `Permission denied (publickey)`

Launcher SSHes as the *local* username to the peer IP. Model B without a
complete `~/.ssh/config` map fails here; model A never hits it.

**Fix (stay on two users):** `Host` blocks per §2.4 for **both** fabric
IPs (and reverse map on worker if needed). Verify
`ssh 10.10.0.2 whoami` → worker account, no prompt.

**Fix (less headwind long-term):** convert B → A per §1.1.3 — shared
username + UID, redo `ssh-copy-id`, remove the Host overrides.

### 4.2 UFW blocks the fabric: Ray GCS timeout

```
[ray] GCS health check failed: 10.10.0.1:29501
```

**Fix:** `sudo ufw allow from 10.10.0.0/24 && sudo ufw allow from 10.10.1.0/24`
on both. Reload.

### 4.3 Only one twin has an IP (~half bandwidth)

**Fix:** redo §2.1 for the missing twin. Both twins must be on **different**
/24s. `autodiscover.sh` explicitly errors if you put them on the same subnet:

```
Error: Interfaces X and Y share the same subnet (10.10.0.0/24).
```

### 4.4 Wrong NCCL_IB_GID_INDEX: `ibv_modify_qp failed 101`

Each HCA exposes 4 GIDs. Only index 3 (RoCE v2 + IPv4-mapped) works across
the fabric. Index 0/1 are link-local and can't resolve. Sparkrun's default
detection picks index 1 — wrong.

**Fix:** the patched autodiscover loop in §2.7 requires BOTH `RoCE v2` AND
`0000:0000:...:ffff:*` (IPv4-mapped) — usually lands on index 3. Verify in
the container env log (§3.6).

### 4.5 NCCL falls back to TCP on switchless fabric

Symptom: throughput is TCP-over-mgmt rate (single-digit Gb/s), container log
shows `NET/Socket` not `NET/IB`.

**Fix:** the three `NCCL_*` overrides from §2.7 patch must be in `.env`
and propagated into the container. Check `cat .env` has all three, and
that `docker logs vllm_node | grep NCCL` shows them set.

### 4.6 SSH/rsync much slower than the fabric

SSH crypto CPU-bound on GB10 Arm cores, ~150 MB/s max per stream.

**Fix:** local docker registry (§2.5) for images and `docker save | nc` for
one-shot large transfers. For model shards, the registry path also works
(pull an image with the weights baked in, or use `rsync` over the registry's
NFS mount if you set one up).

### 4.7 Ray cluster: "current node has no GPU available"

Stale Ray placement group from a prior failed cluster launch.

**Fix:**

```bash
~/projects/spark-vllm-docker/launch-cluster.sh stop
sudo pkill -f 'ray::'                  # both nodes
ssh <user>@10.10.0.2 'sudo pkill -f ray::'
# wait 3s, relaunch
```

### 4.8 Worker stuck in `snapshot_download` during vLLM launch

HF cache path not aligned across nodes — `/mnt/ai/cache/hub/hub/models--<repo>/`
must resolve to real content on both sides.

**Fix:** `rsync -aHAX --partial --info=progress2 -e 'ssh -c aes128-gcm@openssh.com'`
the model dir from head to worker once per model.

---

## 5. Cross-node symmetry (quick diff before a run)

eugr's launcher, sparkrun, and NCCL all assume symmetric nodes. Diff these
on both before a first TP=2 run:

| Property | Check | Why |
|---|---|---|
| Username | `whoami` (same both sides, or §2.4 map) | Launcher SSHes as current user |
| Docker group | `groups \| grep docker` | Launcher needs `docker exec` without sudo |
| Docker daemon | `systemctl is-active docker` | Corrupt buildkit cache is a known silent failure (rm `/var/lib/docker/buildkit` + restart) |
| DGX OS | `cat /etc/dgx-release` | Minor skew OK; if troubleshooting collective hangs, OTA-update the older one |
| Driver | `nvidia-smi \| head -5` | Must be same major |
| Container digest | `docker inspect --format '{{.Id}}' vllm-node-tf5:latest` | **Must match byte-for-byte** — stale peer image = silent shape errors at runtime. Registry (§2.5) keeps them in sync |
| MTU | `ip -br link show <iface>` | Both twins MUST be 9000 |
| NVIDIA Container Toolkit | `docker info \| grep -i nvidia` | `nvidia-container-runtime` listed |
| NTP | `timedatectl status` | Ray GCS expects <30 s skew |

One-liner for side-by-side diff (substitute your own hostnames):

```bash
for h in GX_NODE_1 GX_NODE_2; do
  echo "===== $h ====="
  ssh $h 'whoami; id -Gn; systemctl is-active docker;
          cat /etc/dgx-release 2>/dev/null | head -1;
          nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1;
          docker inspect --format "{{.Id}}" vllm-node-tf5:latest 2>/dev/null;
          timedatectl show -p NTPSynchronized --value'
done
```

---

## 6. Run a model

```bash
cd ~/projects/spark-vllm-docker

# pull + sync a model if not already cached on both
HF_HUB_ENABLE_HF_TRANSFER=1 hf download <org>/<model>
rsync -aHAX --partial --info=progress2 \
  -e 'ssh -c aes128-gcm@openssh.com' \
  /mnt/ai/cache/hub/hub/models--<org>--<model>/ \
  <user>@10.10.0.2:/mnt/ai/cache/hub/hub/models--<org>--<model>/

# launch (reads nodes from .env)
./run-recipe.sh <recipe-name> -d --nccl-debug INFO

# wait for health
until [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health)" = "200" ]; do sleep 10; done

# smoke
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<org>/<model>","messages":[{"role":"user","content":"ping"}],"max_tokens":4}'

# bench
uvx --from git+https://github.com/eugr/llama-benchy llama-benchy \
  --base-url http://localhost:8000/v1 --model <org>/<model> \
  --pp 512 --tg 128 --depth 0 2048 8192 --runs 3 --skip-coherence

# stop
./launch-cluster.sh stop
```

---

## 7. References

- eugr's repo (launcher, recipes, Dockerfile):
  <https://github.com/eugr/spark-vllm-docker>
- eugr's networking deep-wiki:
  <https://deepwiki.com/eugr/spark-vllm-docker/7-dgx-spark-networking>
- NVIDIA official dual-Spark playbook:
  <https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/connect-two-sparks/README.md>
- NVIDIA Spark clustering docs:
  <https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html>
- sparkrun networking:
  <https://sparkrun.dev/getting-started/networking/>
- 6-to-8 node forum thread (scaling past 2):
  <https://forums.developer.nvidia.com/t/6x-spark-setup/354399/56>
- spark-arena leaderboard + benchmark protocol:
  <https://spark-arena.com/leaderboard>
