# ASUS Ascent GX10 / NVIDIA DGX Spark — PD Throttle Wedge: Diagnosis & Fix

**Compiled:** 2026-05-18
**Hardware:** ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell, partner-branded DGX Spark)
**Symptom captured:** GPU pinned at 611 MHz / ~13 W / thermal "cap" at ~50 °C

---

## TL;DR

Your symptom is the **canonical PD (Power Delivery) controller firmware wedge** on GB10 hardware. 611 MHz is the literal hardcoded fallback P-state firmware drops into — not a rounding artifact. The "50 °C thermal cap" is misleading: the GPU is in a low P-state, not thermally limited, so it never gets warm.

**Cold-drain reset of the power brick is the confirmed fix** (6+ independent user confirmations). Holding the power button alone is not enough — the PD MCU lives inside the brick and needs to lose rail voltage.

---

## Action Sequence (least invasive → most)

### 1. Software-level recovery (try first, ~10 seconds)

```bash
sudo nvidia-smi -r                        # GPU reset
sudo nvidia-smi -lgc 300,2300             # re-arm clock range
sudo systemctl restart nvidia-persistenced
sudo fwupdmgr get-devices                 # confirm PD/EC/SOC versions
```

Also check driver version. A stale driver (550.x + CUDA 12.4) on GB10 is a known cause of stuck-at-low-power. Upgrade to **580.95.05 + CUDA 13.0 or newer** — fixed at least one confirmed case without any reset.

**Note:** There is **no `nvpmodel`** on Spark/GX10. GB10 is not Jetson. Ignore any Jetson-derived advice.

### 2. Warm reboot

```bash
sudo reboot
```

If the wedge survives a warm reboot, escalate to cold drain.

### 3. Cold drain (the documented fix)

**Critical:** unplug *all* USB-C peripherals, not just the PSU. A misbehaving downstream PD sink (hub, second-port monitor) can re-trigger safety mode the instant you replug.

1. Capture diagnostics first (gone after the drain):
   ```bash
   sudo nvidia-smi -q > pre_reset.txt
   sudo dmesg -T > pre_reset_dmesg.txt
   sudo fwupdmgr get-devices > pre_reset_fw.txt
   ```
2. `sudo shutdown -h now`
3. Unplug the **240 W USB-C PSU from the wall AND from the unit**
4. Unplug **every** USB-C peripheral (hub, monitor, dock)
5. Hold the power button for **~30 seconds** to drain rails
6. Wait **60 s+** (brick capacitors take longer than you'd think)
7. Replug **PSU only** — no hub, no monitor on the second USB-C
8. Power on
9. After POST: `sudo fwupdmgr get-devices` and confirm versions
10. Add peripherals back one at a time after confirming healthy clocks

### 4. Firmware flash to current versions

After successful recovery, lock in the fix:

```bash *(SEE ADDITIONAL NOTES NEAR END)
sudo fwupdmgr refresh && sudo fwupdmgr update
```

On the GX10 specifically, target **ASUS v0104** (released 2026-05-07):
- SOC FW 3.0.6
- BIOS 0104
- EC 2.78.18.3
- TPM 7.2.4.1
- PD 5.7

Download: https://www.asus.com/networking-iot-servers/desktop-ai-supercomputer/ultra-small-ai-supercomputers/asus-ascent-gx10/helpdesk_bios

A parallel LVFS capsule wave bumps PD further to **0x516** (vs 0x507 in v0104) — explicitly targeting USB-C PD stability. After the ASUS-signed flash, run `fwupdmgr refresh && upgrade` again to pick it up.

### 5. Firmware double-flash (if PD bank didn't update)

ASUS documents this for stuck PD updates: Flash → Reboot → Flash → Reboot. Writes both PD EEPROM banks.

### 6. Factory recovery USB

1. Download DGX Spark recovery `.tar.gz` from developer.nvidia.com
2. Build USB: `CreateUSBKey.sh`
3. Boot holding **Esc/Del**
4. Restore Defaults → Restore Factory Keys
5. Boot override to USB → reflash

Docs: https://docs.nvidia.com/dgx/dgx-spark/system-recovery.html

### 7. Field diagnostic + RMA bundle

Collect *before* opening the ASUS ticket:

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
| GPU clock under load | 611 MHz | **~2400–2600 MHz** |
| Package power under load | ~13 W | **~80–100 W** sustained |
| GPU power limit | shows as 30 W or lower in safety mode | **~140 W** |
| SOC power under load | — | **~140 W** |
| Wall power burst | — | up to **240 W** |
| Temps under load | "capped at 50 °C" (P-state artifact) | **70–90 °C** is normal; NVIDIA staff confirm 86 °C is in-spec |
| Qwen-34B throughput | ~30 tok/s | **~61 tok/s** |
| Throttle-check tool threshold | <1400 MHz = flagged | >1400 MHz = clear |

**Verification tool** (community): https://github.com/hoesing/spark-gpu-throttle-check

If `nvidia-smi -q | grep -i power` still shows a 30 W cap after recovery, you're still in safety mode.

---

## Prevent Recurrence

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

**Known triggers:**
- Sleep/resume (most-reported — one user hit the wedge 3× in one month from suspend cycles)
- Crash mid-load
- Firmware update applied without a full power cycle afterward

No firmware version has been reported as a *permanent* fix — even fully-updated units regress. Keep the cold-drain in the playbook.

---

## Important Context

### Firmware tracks
ASUS publishes a **separate firmware track** from NVIDIA's DGX Spark reference. ASUS-signed capsules ≠ NVIDIA-signed capsules. Do not try to flash NVIDIA's capsules on a GX10 — they will not install. Use the ASUS helpdesk download for v0104, or LVFS via `fwupdmgr` (LVFS gets the ASUS-signed payload). ASUS rebases ~1–2 weeks behind NVIDIA's DGX Spark capsules.

### No BMC / no out-of-band recovery
GB10 has no BMC. All flashing is host-side under DGX OS. No EC-reset pinhole.

### Chassis-specific
InsiderLLM's GB10 comparison notes the **ASUS GX10 is the only GB10 unit in their testing that triggers thermal slowdown events** — chassis/cooling design is the weakest of the GB10 partner lineup. Useful framing for RMA: "this is a known ASUS-chassis-specific pattern."

### Escalation
- **NVIDIA Enterprise Support** will redirect GX10 cases to ASUS RMA — skip them, go direct to ASUS.
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
- **Latest LVFS capsule wave (PD 0x516):** [FYI new firmware available](https://forums.developer.nvidia.com/t/fyi-new-firmware-available/368339)
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


*(ADDITIONAL NOTES)

fwupdmgr refresh        # pull latest metadata from LVFS (read-only)
fwupdmgr get-upgrades   # show what's available, install nothing  ← safe peek
fwupdmgr update         # actually flash pending updates
fwupdmgr upgrade        # alias for `update`, same thing


Safer two-step:

sudo fwupdmgr refresh
sudo fwupdmgr get-upgrades       # read what's pending — confirm 0x516 PD shows up
# review output, then if it looks right:
sudo fwupdmgr update

After update completes: full unplug + cold-drain before the next boot — per the doc's recurrence-trigger #3, 
a reboot alone after firmware flash can leave the bug in place.