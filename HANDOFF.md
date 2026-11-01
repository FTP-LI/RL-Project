# RL Project development handoff

Updated: 2026-11-01. This repository contains the XGO-Duck research, official source archive and first board bring-up record, plus the earlier XGO-Lite2 planning project. Start here after moving to another computer.

## Current objective

The user owns an XGO-Duck kit with an Arduino UNO Q, the robot expansion board and servos. The intended robot should walk using the official baseline and later trained policies, look toward the user while chatting, and understand spoken motion commands. USB / USB-C peripherals are preferred; the MIPI camera path is deferred.

Mechanical assembly and servo ID assignment are user-confirmed. The UNO Q has been connected, updated and flashed with runtime `b401d85`; the web UI, 2 Mbps Bridge and all 15 expected servo IDs were verified, with feedback around 100 Hz. **IMU initialization remains unresolved; servo mechanical zeros, standing and walking have not been validated.** Eight ONNX models loaded, but inference stayed at 0 Hz because the IMU did not initialize. Training, camera streaming and voice conversation remain untested.

Read the [2026-11-01 bring-up record](XGO-Duck/bringup/2026-11-01/README.md) before resuming work. Its last verified robot state is `shadow` with torque disabled. The board was disconnected when the record was archived, so saved telemetry is historical rather than live.

## Restore on another computer

```bash
git clone https://github.com/FTP-LI/RL-Project.git
cd RL-Project
git status --short
git log -1 --oneline
```

A normal clone includes the four pinned XGO-Duck source trees, their CAD/mesh/model files, four ZIP snapshots, research documents and the XGO-Lite2 planning archive. No Git submodules or Git LFS download is required. Do not use a sparse checkout if a complete copy is wanted.

Run the PowerShell verification block in [Official resources: Restore on another computer](XGO-Duck/official_resources/README.md#restore-on-another-computer). It checks every archived source file, file manifest and ZIP. The expected result is **585 source files and 4 ZIP snapshots**. That page also provides offline ZIP extraction instructions.

The additional `XGO-Duck/bringup/2026-11-01/` directory contains the actual deployed `b401d85` source ZIP, diagnostic sketch, logs and state excerpts. Its separate `manifest.json` and `SHA256SUMS` cover these files. Keep this deployment record distinct from the original four-source archive and its earlier runtime baseline.

Git restores the project files, not installed environments or an entire workstation. Reinstall Git, the development tools, the training environment and the UNO Q tooling as needed. Python packages and other external dependencies still require installation; the archived source does not contain all dependency wheels or system images.

## What is preserved

| Area | Contents and entry point |
| --- | --- |
| XGO-Duck overview | [README](XGO-Duck/README.md), [project metadata](XGO-Duck/project.json) |
| Official hardware | BOM, assembly PDF, STL parts, expansion-board PCB and schematic files in `XGO-Duck/official_resources/xgoduck_hardware/` |
| Official runtime baseline | UNO Q application, MCU sketch and five supplied ONNX policies in `XGO-Duck/official_resources/xgoduck_runtime_arduino/` |
| Official training | XGO-Duck models, meshes, actuator model, task code, `pyproject.toml`, `uv.lock` and vendored BAM in `XGO-Duck/official_resources/xgoduck_rl/` |
| Audio and video preview | A separate pinned runtime branch in `XGO-Duck/official_resources/runtime_media_preview/`; keep it distinct from the runtime baseline |
| Provenance and restoration | [Source manifest](XGO-Duck/official_resources/source_manifest.json), per-file SHA-256 manifests, four original ZIPs and their checksums |
| Actual board bring-up | [2026-11-01 report](XGO-Duck/bringup/2026-11-01/README.md), deployed `b401d85` ZIP, version/status records, temporary IMU probe and key logs |
| Reinforcement learning research | [Deployment plan](XGO-Duck/docs/rl_deployment_plan.md), [policy contract report](XGO-Duck/docs/policy_contract_report.json) |
| Accessories and geometry | [Hardware notes](XGO-Duck/docs/hardware_and_accessories.md), [camera and voice selection](XGO-Duck/docs/camera_voice_selection.json), [OV2735 section drawing](XGO-Duck/docs/assets/ov2735_fit_screening.png) |
| Earlier XGO-Lite2 work | [Project overview](XGO-Lite2/README.md), stage documents, metadata and the 2026-09-11 planning snapshot |

The official source trees contain 119,819,182 bytes and the four ZIPs contain 82,757,466 bytes. Their combined size is **202,576,648 bytes**, excluding project notes and Git history.

| Source | Pinned commit |
| --- | --- |
| Hardware | `b5e9ff16a98f7bbe5c146aad29ad38f0dde30a89` |
| Runtime master | `8837d5a3bc7cf3746f6da2b55c7b9191026b9838` |
| Training | `fa621fc77ad9a05d69da13270e86118bdd3603fe` |
| Runtime media preview | `44b2ac852cc3b03aca0de33996a6b4ff3cd0cab4` |

Upstream sources retain their original bytes, languages and licensing notices. They are ordinary directories in this repository, not separate Git checkouts. Local `.upstream_git/` metadata backups are ignored and are not required for file restoration. Preserve the pinned archive when creating a future development copy.

## Current board bring-up findings

- Board tools recorded after update: App CLI 0.14.0, Arduino CLI 1.5.1, router 0.11.0, Linux `7.0.0-g122c2c22d838`. The global Arduino Q core is 1.1.0, while the deployed sketch intentionally retains its upstream 1.0.0 profile.
- Official `prepare-bridge.py`, `router-2m.sh` and `reuse-toolchain.py` were used. The router runs at 2000000 baud. Python numpy 2.2.6 and onnxruntime 1.22.1 were installed and imported successfully in a network-disabled container.
- A post-update OpenOCD write timeout was recovered with the board's `arduino-reset` and a successful retry. Its underlying cause was not proven; no OpenOCD configuration or runtime profile was changed to suppress it.
- IMU responds at 0x6A at both 100 kHz and 400 kHz, with WHO_AM_I 0x05 and revision 0x7C. Its reset result at 0x4D is 0x00 instead of the required 0x80 after both 20 ms and 500 ms. A complete power cycle did not restore initialization in the original runtime. Hardware failure versus driver compatibility has not been established.
- A temporary probe was compiled and tested, then the original runtime was restored. No servo centers were written and no motion was enabled. The matching mechanical-zero reference set is still missing.

Use the new runtime's Calibration page semantics: enter to unload, place one joint at a verified mechanical center, then use its **Set center & hold** button. Do not assume the earlier archive's `zero_pos.json` workflow applies to this deployed revision. Refer to the bring-up record for the remaining steps and evidence.

## Latest camera and voice findings

The user wants both face-directed conversation and spoken motion commands. The following are **candidates, not purchased or approved final hardware**. Acceptance of an external audio enclosure/backpack is still undecided.

| Component | Candidate and dimensions | Integration status |
| --- | --- | --- |
| Camera | Waveshare **OV2735 2MP USB Camera (A)**; 25 x 25 mm PCB, 22.1 mm overall depth, USB-C, 1080p30 | Preferred over the earlier 38 x 38 mm OV2710 candidate. PCB section screening is promising, but the eye pieces need modification and a bracket. Lens, connector, cable, mounting and moving-servo clearance remain unverified. |
| Audio with onboard processing | Seeed **ReSpeaker Lite 2-Mic Array**, without XIAO ESP32S3, using USB firmware; 35 x 86 mm | Preferred for onboard echo cancellation and simultaneous USB capture/playback. No confirmed internal mounting location; a protected external cover or backpack may be needed. |
| Smaller audio alternative | Waveshare **USB TO AUDIO**; case 53 x 23.5 x 14.6 mm, excluding the USB-A plug | Easier size candidate, but no manufacturer-confirmed onboard AEC was found. Continuous conversation would need software echo cancellation and testing; turn-taking is simpler. |
| Small speaker candidate | Adafruit **Mini Oval Speaker 3923**, 30 x 20 x 5 mm, 8 ohm / 1 W | Needs connector adaptation and amplifier output limited to its 1 W rating. Mounting, acoustic cavity and usable volume remain unverified. |

Manufacturer URLs and dimensional evidence are recorded in [camera_voice_selection.json](XGO-Duck/docs/camera_voice_selection.json). The original OV2710 analysis is retained as an earlier candidate, not the latest recommendation. All fit drawings are preliminary geometric screening, not an approved assembly.

The intended connection is camera plus one recording/playback USB audio device through a powered USB-C hub to the UNO Q. Camera-side USB-C does not remove the need to check the hub, USB host role, power budget and plug clearance. ReSpeaker microphone ports need acoustic openings; burying the board in a sealed body is not a finished audio design.

The archived media branch implements camera preview/stills, microphone recording/streaming, playback and Doubao text-to-speech. It does **not** provide the complete face-tracking conversational system. Its current audio backend selects the first USB sound card, assumes a `Speaker` mixer control and serializes capture/playback. Device identity, mixer control, sample rate/channel handling, full-duplex audio, speech recognition, conversation and bounded motion-command routing still need development. API credentials belong in a private board-side configuration, not Git.

## Continue development

1. **Verify restored files**, then read the project overview and deployment plan. The old workstation observation was an RTX 5070 with about 12 GB VRAM and Ubuntu 24.04 under WSL2; this is not a requirement or a claim about the new computer.
2. **Resolve IMU initialization and remaining hardware checks**: confirm expansion-board revision, actual sensor/power circuit, UNO Q RAM, servo model/firmware, battery/charger and assembly quality. All 15 expected servo IDs have responded, but physical ID placement and mechanical zeros still need verification. Check applicability of the old-board U3/U4/U5 correction against the actual revision.
3. **Establish the official motion baseline** after IMU samples and inference are verified. Obtain the matching mechanical-zero references, calibrate one joint at a time with the robot supported, then test default pose and low-command walking. Record the deployed runtime revision and calibration observations; do not substitute another robot's offsets or an older calibration workflow.
4. **Set up training in Linux or WSL2** from the pinned training source and lockfile. The archive declares Python 3.12, `mjlab==1.3.0`, `warp-lang==1.12.0` and `torch==2.9.1`; follow the deployment plan and validate GPU access before a training run.
5. **Resolve the audio enclosure tradeoff and prototype USB peripherals**. Verify capture/playback and echo behaviour before finalizing printed mounts. Keep perception and dialogue outside the motion-control loop.
6. **Implement face/head tracking and speech interaction**, then map approved motion intents to bounded high-level commands with a timeout. Recognition or model-generated text must not directly issue arbitrary servo commands.

Training and deployment must reconcile the default joint-pose mismatch: archived walking training uses +/-24 degrees, runtime master +/-22 degrees, and the media preview +/-23 degrees. Equal observation/action dimensions do not establish policy compatibility. See the deployment plan for joint order, units and model contracts.

## Files and state that are not restored by this repository

- Installed Python/WSL/CUDA environments, board operating-system images, credentials and unrecorded physical calibration.
- Temporary web downloads, manufacturer CAD experiments and Python packages under the original machine's `.codex/scratch/` directories. The conclusions, source URLs and final screening image are preserved in this repository.
- Upstream Git histories; clone an official upstream separately if that history is needed.
- The earlier XGO-Lite2 external STEP asset referenced at `D:\BaiduNetdiskDownload\XGO-lite2总装2.STEP`; it was not part of the XGO-Duck archive and is not included in this Git repository.
- The H7MS-Core-A1 project, which has its own repository linked from the XGO-Lite2 overview.

## Prompt for the next development session

> Read `HANDOFF.md`, `XGO-Duck/README.md`, `XGO-Duck/project.json` and `XGO-Duck/bringup/2026-11-01/README.md` first. Verify both the original official archive and the separate bring-up manifest. Mechanical assembly and servo IDs are complete; runtime b401d85 is deployed, with 2 Mbps Bridge and 15/15 servo feedback. IMU reset-result verification fails (0x4D=0x00, expected 0x80), including after a full power cycle. The original runtime was restored and torque was disabled. Resolve the IMU, obtain mechanical-zero references and calibrate before motion tests. Training and audio/video remain untested. The user wants face-directed conversation and spoken motion commands through USB peripherals; component choices and enclosure tradeoffs remain unresolved. Preserve the pinned official archives and distinguish historical observations from new measurements.
