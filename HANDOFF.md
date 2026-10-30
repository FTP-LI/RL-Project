# RL Project development handoff

Updated: 2026-10-30. This repository contains the XGO-Duck research and official source archive, plus the earlier XGO-Lite2 planning project. Start here after moving to another computer.

## Current objective

The user owns an XGO-Duck kit with an Arduino UNO Q, the robot expansion board and servos. The intended robot should walk using the official baseline and later trained policies, look toward the user while chatting, and understand spoken motion commands. USB / USB-C peripherals are preferred; the MIPI camera path is deferred.

The project is at the archive, hardware inventory and integration-planning stage. No robot has been connected, flashed, calibrated or tested by this task. No training run, deployed policy, camera stream or voice conversation has been validated on the hardware.

## Restore on another computer

```bash
git clone https://github.com/FTP-LI/RL-Project.git
cd RL-Project
git status --short
git log -1 --oneline
```

A normal clone includes the four pinned XGO-Duck source trees, their CAD/mesh/model files, four ZIP snapshots, research documents and the XGO-Lite2 planning archive. No Git submodules or Git LFS download is required. Do not use a sparse checkout if a complete copy is wanted.

Run the PowerShell verification block in [Official resources: Restore on another computer](XGO-Duck/official_resources/README.md#restore-on-another-computer). It checks every archived source file, file manifest and ZIP. The expected result is **585 source files and 4 ZIP snapshots**. That page also provides offline ZIP extraction instructions.

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
2. **Inventory the actual robot**: UNO Q RAM variant, expansion-board revision, servo model/count, battery/charger, printed parts, bearings and cables. Check whether the official old-board U3/U4/U5 wiring correction applies before connecting servos.
3. **Establish the official motion baseline** with verified power, individual servo IDs, calibration and supported testing. Preserve board-specific `zero_pos.json` separately when it is created.
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

> Read `HANDOFF.md`, `XGO-Duck/README.md`, `XGO-Duck/project.json` and the linked deployment/accessory notes first. Verify the official archive. Continue the XGO-Duck project from its recorded state: UNO Q, expansion board and servos are owned; motion, training and audio/video have not been tested on hardware. The user wants face-directed conversation and spoken motion commands through USB peripherals. Camera/audio models remain candidates, and the enclosure tradeoff is unresolved. Preserve the pinned official archive, keep secrets out of Git and distinguish researched compatibility from measured results.
