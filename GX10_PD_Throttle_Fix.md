# NVIDIA GB10 (DGX Spark) — PD Throttle Wedge: Diagnosis & Fix

> **Status:** verified end-to-end procedure (was: draft). Cold drain → flash → second cold drain workflow executed on ASUS Ascent GX10 (gx10) 2026-05-19, took the box from 611 MHz / 16 W wedged to 2180 MHz / 93 W / 56 bf16 TFLOPS healthy. ESRT version-readback caveat (§6) was discovered during that verification and is captured here so future operators don't mistake it for a flash failure.

**Last updated:** 2026-05-19 (PD 0x507 flash + post-flash cold drain procedure verified on ASUS Ascent GX10)
**Applies to:** NVIDIA GB10 Grace Blackwell — both NVIDIA-reference DGX Spark and partner-branded variants (ASUS Ascent GX10, Dell partner GB10, etc.)
**Symptom captured:** GPU pinned at 611 MHz / ~13 W / thermal "cap" at ~50 °C

---

## TL;DR

The symptom — **611 MHz pinned clock, ~13 W power draw, ~50 °C temperature cap under load** — is the canonical USB-C PD (Power Delivery) controller firmware wedge on GB10 hardware. 611 MHz is the literal hardcoded fallback P-state firmware drops into, not a rounding artifact. The temperature "cap" is misleading: the GPU is in a low P-state, not thermally limited, so it never gets warm enough to actually throttle.

**Cold-drain reset of the power brick is the confirmed primary fix** (6+ independent user confirmations, plus our own 2026-05-19 gx10 confirmation). Holding the power button alone is not enough — the PD MCU lives inside the brick and needs to lose rail voltage. **After the unwedge, flash the latest vendor-signed PD firmware** to reduce recurrence, then do a *second* cold drain so the new firmware activates from EEPROM.

---

## Vendor / firmware tracks (important context)

Each GB10 vendor publishes a **separate firmware track**. The capsules they sign do not interchange — a flash from a different vendor's track will refuse to install.

| Vendor | PD target (current stable, 2026-05-19) | Source |
|---|---|---|
| **ASUS Ascent GX10** | **`0x00000507`** (LVFS release 133405, ASUS-tested 2026-01-30) | LVFS via `fwupdmgr`, or ASUS helpdesk |
| NVIDIA DGX Spark reference | Track tag varies; check `fwupdmgr get-upgrades` | NVIDIA DGX OS update channel |
| Dell partner GB10 | Dell-signed capsule track | Dell update channel |

The `0x507` target in this doc is **ASUS-specific**. On another vendor's chassis, `fwupdmgr get-upgrades` will name the equivalent PD capsule — flash whatever the **vendor-signed LVFS metadata** offers, not a hardcoded version. ASUS typically rebases ~1–2 weeks behind NVIDIA's DGX Spark reference capsules.

A parallel LVFS capsule wave for PD `0x516` exists in some discussions, but as of 2026-05-19 it is **not visible** for the ASUS GX10 — only `0x507` is the LVFS-current stable for that chassis. If a newer capsule appears in `get-upgrades` later, that's the new target.

---

## Action Sequence (least invasive → most)

### 1. Software-level recovery (try first, ~10 seconds)

```bash
sudo nvidia-smi -r                        # GPU reset (often fails on Spark: held by Xorg/GDM)
sudo nvidia-smi -lgc 300,2300             # re-arm clock range
sudo systemctl restart nvidia-persistenced
sudo fwupdmgr get-devices                 # confirm PD/EC/SOC versions
```

Also check driver version. A stale driver (550.x + CUDA 12.4) on GB10 is a known cause of stuck-at-low-power. Upgrade to **580.95.05 + CUDA 13.0 or newer** — fixed at least one confirmed case without any reset.

**Caveats observed in practice (gx10, 2026-05-19):**
- `nvidia-smi -r` reliably fails on Spark with `"In use by another client"` — `Xorg`/`gnome-shell` hold `/dev/nvidia0` from the GDM session. Stopping the display server to satisfy `-r` is not worth it; the wedge is in the PD MCU, not in software state.
- A re-armed clock + persistenced restart **does not** clear a true PD wedge. Re-running preflight under load will show the same 611 MHz / ~16 W signature as before.
- If preflight passes after this rung, the original symptom was likely a transient driver state, not a PD wedge.

**Note:** There is **no `nvpmodel`** on Spark/GB10. GB10 is not Jetson. Ignore any Jetson-derived advice.

### 2. Warm reboot

```bash
sudo reboot
```

If the wedge survives a warm reboot, escalate to cold drain. A warm reboot keeps the 240 W brick's rails powered the entire time, so the PD MCU never loses state.

### 3. Cold drain (the documented fix)

**Critical:** unplug *all* USB-C peripherals, not just the PSU. A misbehaving downstream PD sink (hub, second-port monitor) can re-trigger safety mode the instant you replug.

1. Capture diagnostics first (gone after the drain):
   ```bash
   sudo nvidia-smi -q > pre_reset_smi.txt
   sudo dmesg -T    > pre_reset_dmesg.txt     # needs sudo; kernel.dmesg_restrict=1
   sudo fwupdmgr get-devices > pre_reset_fw.txt
   journalctl -b -0 > pre_reset_journal.txt   # user-readable, no sudo needed
   ```
   Also useful: a preflight run under load *before* the drain, as the "before" datapoint for the fix.
2. `sudo shutdown -h now`
3. Unplug the **240 W USB-C PSU from the wall AND from the unit**
4. Unplug **every** USB-C peripheral (hub, monitor, dock — on both Type-C ports)
5. Hold the power button for **~30 seconds** to drain rails
6. Wait **60 s+** (brick capacitors take longer than you'd expect)
7. Replug **PSU only** — no hub, no monitor on the second USB-C
8. Power on
9. **Fabric link wait:** on a cluster with a peer node over CX-7/NIC, the fabric may take 30–90 s after the OS boots before peer-to-peer reachability returns. `ping <peer>` initially returns `No route to host` — that's the local interface still negotiating, not a failure. Wait and re-ping.
10. After POST + fabric up: re-run preflight under load. Healthy = `≥1400 MHz` mean clock and `≥40 W` mean power for ≥30 s. Typical post-drain numbers on GB10: **~2200 MHz / ~93 W / ~56 bf16 TFLOPS**.
11. `sudo fwupdmgr get-devices` and confirm versions (subject to the ESRT readback caveat in §6), then add peripherals back one at a time

### 4. Firmware flash to current vendor-signed version

After successful recovery, lock in the fix by flashing the latest **vendor-signed** PD capsule. **Safer two-step:**

```bash
sudo fwupdmgr refresh                  # pull latest metadata from LVFS (read-only)
sudo fwupdmgr get-upgrades             # show what's available, install nothing  ← safe peek
# review output — confirm the offered version matches the expected vendor track,
# matches what a healthy peer node in the same vendor family already runs,
# and is "Signed Payload" / "Tested by trusted vendor"
sudo fwupdmgr update                   # actually flash pending updates
```

**Example: ASUS Ascent GX10 (verified 2026-05-19, gx10 from 0x001 → 0x507)**

`get-upgrades` showed:

```
└─GX10 USB-C PD FW Controller Update:
      New version:      0x00000507
      Remote ID:        lvfs
      Release ID:       133405
      Summary:          GX10 USB-C PD FW Firmware Update
      Size:             1.1 MB
      Created:          2025-12-09
      Tested by Asus:   2026-01-30
      Release Flags:    • Trusted metadata
                        • Is upgrade
                        • Tested by trusted vendor
      Description:      Isolated the SBU mux when the U4_RDY signal is low.
                        Added a workaround to the MSI Dongle EXIT_DP_MODE issue.
```

`fwupdmgr update --assume-yes --no-reboot-check` ran in ~30 s on the device, reported `Successfully installed firmware`. The `--no-reboot-check` flag is important so the post-flash power cycle is the *cold drain* in step 5, not an automatic warm reboot (which can leave the new firmware un-activated — see §5).

**Other vendors:** target the version that **`get-upgrades` reports as available for that vendor's GUID**. Do **not** try to flash NVIDIA's reference capsules on a partner chassis or vice versa — they will refuse to install.

### 5. Second cold drain (post-flash) — required, not a reboot

After the flash, **do not just reboot.** A warm reboot keeps the 240 W brick rails powered, so the PD MCU never loses state and never re-loads the new firmware image from EEPROM. The flash will sit in the controller's EEPROM but never become active.

Same physical procedure as §3:

1. `sudo shutdown -h now`
2. Unplug PSU from wall **and** from unit
3. Unplug all USB-C peripherals
4. Hold power button 30 s
5. Wait 60 s+
6. Replug PSU only
7. Power on

After this second drain, the PD MCU cold-boots and loads the freshly written EEPROM image, which is when the new firmware becomes live.

**Self-handshake on first post-flash boot (observed 2026-05-19, gx10):** on the *first* power-on after the post-flash drain, the system may briefly power on, power off, and power back on under its own control before settling. This appears to be the new PD firmware renegotiating the USB-C contract during initial handshake and is **not a failure** — just give it a couple of minutes to stabilize before logging in or running `fwupdmgr`. Subsequent boots are normal.

### 6. Verify the flash took — and the ESRT readback caveat

```bash
sudo fwupdmgr get-devices | grep -B6 -A4 "<PD-device-GUID>"   # ASUS: fe75bb1c-5ccc-4936-b603-cc7cf945dc30
```

**Caveat (observed 2026-05-19, gx10 ASUS GX10):** on first-flash, `Current version` in `fwupdmgr get-devices` can **stay at the pre-flash value** (e.g. `0x00000001`) even after a successful flash + cold drain. The companion field `Update State: Success` is the trustworthy signal — the misleading version readback comes from the UEFI System Resource Table (ESRT) which fwupdmgr queries via the capsule-on-disk path. On Spark, UEFI does not always refresh the ESRT field after a PD-controller-only flash, so the version stays at the original UEFI snapshot. The live PD firmware revision lives inside the PD MCU EEPROM and is not directly exposed.

**Trust functional behavior, not the version string. Expected progression of trustworthy signals:**

| Stage | `Update State` | `Current version` (ESRT) | preflight load test |
|---|---|---|---|
| Pre-flash (wedged) | (previous flash record) | `0x00000001` (whatever was there before) | FAIL — 611 MHz / ~16 W |
| Just after `fwupdmgr update`, **before** post-flash drain | `Needs reboot` | unchanged | (don't run yet — system still on staged firmware) |
| **After** post-flash cold drain | **`Success`** | may **still** read pre-flash value (ESRT quirk) | **PASS** — ~2200 MHz / ~93 W |

Concretely, the signals that prove the flash took:

- ✅ `Update State: Success` for the PD device (post-drain)
- ✅ Preflight `PASS` under load (clock ≥1400, power ≥40 W, ideally 2200/93 W)
- ✅ Wedge **does not return** on the second boot after the post-flash cold drain (or on a third, fourth boot — wedge would re-trigger on every cold boot if PD firmware were still the old image)
- ✅ Historic SW Power Cap counter (`nvidia-smi -q -d PERFORMANCE | grep "SW Power Capping"`) drops from multi-thousand-second accumulation to near zero after the drain, and **does not re-accumulate** during a 30 s preflight load test
- ⚠️ `Current version` may still read the old value — do **not** treat that alone as failure
- ⚠️ `fwupdmgr get-upgrades` correctly reports `No updates available` for the PD device after the flash — the LVFS-side reconciliation works even though the ESRT readback is stale

If preflight FAILs post-flash-drain, *then* the flash genuinely didn't take and §7 applies. Don't run the double-flash just because the version string didn't move.

### 7. Firmware double-flash (only if PD bank actually didn't update)

Vendor-documented for stuck PD updates: Flash → Reboot → Flash → Reboot (or Flash → cold drain → Flash → cold drain for extra safety). Writes both PD EEPROM banks. Only escalate here if functional behavior is still wedged after §5+§6.

### 8. Factory recovery USB

1. Download DGX Spark recovery `.tar.gz` from developer.nvidia.com
2. Build USB: `CreateUSBKey.sh`
3. Boot holding **Esc/Del**
4. Restore Defaults → Restore Factory Keys
5. Boot override to USB → reflash

Docs: https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html

### 9. Field diagnostic + RMA bundle

Collect *before* opening the vendor ticket:

```bash
sudo apt install dgx-spark-fieldiag      # disable Secure Boot first
sudo ./partnerdiag --field               # ~30 min, RMA-qualifying report

sudo nvidia-bug-report.sh                # produces nvidia-bug-report.log.gz
sudo dmesg -T > dmesg.txt
journalctl -b -0 > journal.txt
sudo fwupdmgr get-devices > fw.txt
nvidia-smi -q > smi.txt
```

---

## Healthy Baseline (Post-Recovery Verification)

| Metric | Wedged state | Healthy GB10 |
|---|---|---|
| GPU clock under load | 611 MHz | **~2200–2600 MHz** |
| Package power under load | ~13 W | **~80–100 W** sustained |
| GPU power limit | shows as 30 W or lower in safety mode | **~140 W** |
| SOC power under load | — | **~140 W** |
| Wall power burst | — | up to **240 W** |
| Temps under load | "capped at 50 °C" (P-state artifact) | **70–90 °C** is normal; NVIDIA staff confirm 86 °C is in-spec |
| Qwen-34B throughput | ~30 tok/s | **~61 tok/s** |
| bf16 GEMM TFLOPS (preflight) | ~18 | **~55–60+** |
| Throttle-check tool threshold | <1400 MHz = flagged | >1400 MHz = clear |

**Verification tool** (community): https://github.com/hoesing/spark-gpu-throttle-check

If `nvidia-smi -q | grep -i power` still shows a 30 W cap after recovery, you're still in safety mode.

---

## Prevent Recurrence

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
sudo systemctl enable --now nvidia-persistenced
# If GNOME:
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type "nothing"
```

**Known recurrence triggers:**
- Sleep/resume cycles (most-reported — one user hit the wedge 3× in one month from suspend)
- Crash mid-load
- Firmware update applied without a full power cycle afterward (warm reboot is not enough)

No firmware version has been reported as a *permanent* fix — even fully-updated units regress. Keep the cold-drain in the playbook.

---

## Important Context

### Vendor firmware tracks (recap)
NVIDIA, ASUS, and Dell publish **separate, signed** firmware tracks for the same GB10 silicon. Cross-vendor capsules will not install. Always flash whatever your vendor's LVFS metadata offers as the current target. ASUS rebases ~1–2 weeks behind NVIDIA reference; Dell similarly lags.

### No BMC / no out-of-band recovery
GB10 has no BMC. All flashing is host-side under DGX OS / Ubuntu. No EC-reset pinhole.

### Chassis-specific (ASUS GX10 caveat)
InsiderLLM's GB10 comparison notes the **ASUS GX10 is the only GB10 unit in their testing that triggers thermal slowdown events** — chassis/cooling design is the weakest of the GB10 partner lineup. Useful framing for RMA: "this is a known ASUS-chassis-specific pattern."

### Escalation
- **NVIDIA Enterprise Support** will redirect partner-chassis cases to the partner's RMA — skip them, go direct to the vendor (ASUS / Dell / etc.).
- **ASUS RMA portal:** https://www.asus.com/us/supportonly/gx10/ (filed under "MiniPC / Desktop AI Supercomputer," not ROG or ProArt).
- **Heads up:** active forum thread "Avoid ASUS GX10s If You Want Any Support" reports 5+ weeks for ConnectX-7 RMAs, ASUS requires ship-in troubleshooting (no unit-swap). Plan accordingly.

---

## Primary Sources

### Direct symptom matches (NVIDIA forums)
- **611 MHz exact match:** [Power delivery bug: How I'm catching it early](https://forums.developer.nvidia.com/t/power-delivery-bug-how-im-catching-it-early/366944) — Apr 18 2026, user captured CLOCK THROTTLED at 611 MHz under 96% load
- **14 W match + 6 confirmations of cold-drain fix:** [DGX Spark Performance Degradation - GPU Power Draw Issue](https://forums.developer.nvidia.com/t/dgx-spark-performance-degradation-gpu-power-draw-issue/361294)
- **533 MHz / 14 W match:** [DGX Spark GPU power usage cap at 14W](https://forums.developer.nvidia.com/t/dgx-spak-gpu-power-usage-cap-at-14w/363487)
- **Permanent 30 W safety mode / PD negotiation failure:** [ASUS GX10 / DGX Spark Permanent Power Throttle at 30W](https://forums.developer.nvidia.com/t/asus-ascent-gx10-dgx-spark-permanent-power-throttle-at-30w-safety-mode-pd-firmware-negotiation-failure/355255)
- **Stuck at 5 W, fixed by driver upgrade:** [GB10 GPU stuck at 5W and 0% utilization](https://forums.developer.nvidia.com/t/dgx-spark-gb10-gpu-is-stuck-at-5w-power-and-0-utilization-even-after-all-nvidia-firmware-updates/356426)
- **Spark hangs requiring hard reset:** [Spark hangs / hard-reset required](https://forums.developer.nvidia.com/t/spark-hangs-requires-a-hard-reset-physically-unplugging/358951)

### Diagnostic tools
- [GPU PD Throttle Check Tool](https://forums.developer.nvidia.com/t/gpu-pd-throttle-check-tool/362737) — community diagnostic, flags <1400 MHz under load
- [hoesing/spark-gpu-throttle-check (GitHub)](https://github.com/hoesing/spark-gpu-throttle-check)
- [Local diagnostic CLI for DGX Spark](https://forums.developer.nvidia.com/t/stop-guessing-why-your-dgx-spark-is-slow-i-built-a-local-diagnostic-cli/367856)

### Firmware
- **ASUS GX10 BIOS download (v0104, 2026-05-07):** https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/helpdesk_bios
- **ASUS firmware update FAQ:** https://www.asus.com/us/support/faq/1056213/
- **ASUS firmware discussion (v0103 deltas):** [ASUS Ascent GX10 firmware update](https://forums.developer.nvidia.com/t/asus-ascent-gx10-firmware-update/364160)
- **Earlier LVFS capsule wave discussion (PD 0x516, not yet on ASUS stable as of 2026-05-19):** [FYI new firmware available](https://forums.developer.nvidia.com/t/fyi-new-firmware-available/368339)
- **ServeTheHome firmware track explainer:** https://www.servethehome.com/nvidia-dgx-spark-and-dell-partner-gb10-firmware/

### Official NVIDIA documentation
- [DGX Spark OS & Component Update](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html)
- [DGX Spark System Recovery](https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html)
- [DGX Spark Maintenance & Troubleshooting](https://docs.nvidia.com/dgx/dgx-spark/maintenance-and-troubleshooting.html)
- [DGX Spark Field Diagnostics](https://nvidia.custhelp.com/app/answers/detail/a_id/5767/)

### Press / community context
- [Tom's Hardware: DGX Spark Review (clocks/power baseline)](https://www.tomshardware.com/pc-components/gpus/nvidia-dgx-spark-review)
- [Tom's Hardware: Carmack throttle / 100 W cap reports](https://www.tomshardware.com/tech-industry/semiconductors/users-question-dgx-spark-performance)
- [Tom's Hardware: idle-power firmware update saves 32%](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidia-dgx-spark-update-cuts-idle-power-by-32-percent-or-more-hot-plug-detection-on-connectx-nic-makes-for-a-more-efficient-ai-workstation)
- [InsiderLLM: GB10 boxes compared — ASUS GX10 thermal flag](https://insiderllm.com/guides/gb10-boxes-compared/)
- [ASUS support quality warning thread](https://forums.developer.nvidia.com/t/avoid-asus-gx10s-if-you-want-any-support/369230)
- [ggerganov llama.cpp DGX Spark numbers](https://x.com/ggerganov/status/1978106631884828843)

---

## Appendix: fwupdmgr cheat sheet

```bash
sudo fwupdmgr refresh           # pull latest metadata from LVFS (read-only)
sudo fwupdmgr get-upgrades      # show what's available, install nothing  ← safe peek
sudo fwupdmgr get-devices       # current state of all devices fwupd knows about
sudo fwupdmgr get-history       # past flash attempts and their outcomes
sudo fwupdmgr update            # actually flash pending updates (interactive)
sudo fwupdmgr update --assume-yes --no-reboot-check
                                # non-interactive, skip auto-reboot prompt
                                # (so the post-flash power cycle can be a cold drain, not a warm reboot)
sudo fwupdmgr upgrade           # alias for `update`
```

**Recommended PD-flash workflow:**

```bash
sudo fwupdmgr refresh
sudo fwupdmgr get-upgrades                       # review — confirm vendor-signed PD capsule, trusted metadata
sudo fwupdmgr update --assume-yes --no-reboot-check
# do NOT warm-reboot — instead:
sudo shutdown -h now
# physical: unplug PSU, all USB-C, hold power 30s, wait 60s, replug PSU only, boot
# then verify with preflight under load, not just the version string (see §6 caveat)
```
