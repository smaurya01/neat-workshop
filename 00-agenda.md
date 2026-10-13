# Building Edge AI Video Apps with SiMa Neat on Modalix — Workshop

**Duration:** about 2.5 hours, hands-on  
**Platform:** Modalix DevKit, board software 2.1.3, Neat SDK 2.1.3.1, Neat Library 0.5.0  
**Language:** C++ (everything here also has a Python equivalent, `pyneat`)

By the end of the workshop you will have:

1. Compiled two YOLO26 models yourself, one in **INT8** and one in **BF16**.
2. Built and run a **single-stream** app: one video source → one model → live video plus boxes in Neat Insight.
3. Built and run a **dual-stream** app: two video sources → two models (detection and pose) → two live outputs.

---

## Agenda

| Time | Duration | Session | Material |
|---|---|---|---|
| 0:00 – 0:45 | 45 min | **Software installation and introduction** | [Installation checklist](#session-0--installation-checklist-45-min) below |
| 0:45 – 1:05 | 20 min | **Neat C++ API introduction** | [01-neat-cpp-api.md](01-neat-cpp-api.md) |
| 1:05 – 1:35 | 30 min | **Model compilation: INT8 and BF16** | [02-model-compilation.md](02-model-compilation.md) |
| 1:35 – 1:55 | 20 min | **App 1: one source, one model, one stream** | [03-app1-single-stream.md](03-app1-single-stream.md) |
| 1:55 – 2:25 | 30 min | **App 2: two sources, two models (detection and pose), two streams** | [04-app2-dual-stream.md](04-app2-dual-stream.md) |
| 2:25 – end | rest | **Questions and answers** | |

> **Instructor tip:** an INT8 compile takes about 7 minutes and a BF16 compile about 9 minutes (measured on a 14-core, 16 GB host). Get attendees to
> Step 3 (the compile) within about 10 minutes of session 2 and explain while it runs. Keep the prebuilt archives
> (see [Prebuilt fallback](#prebuilt-fallback)) on hand so no one waits on a compile to start App 1.

---

## What runs where

```text
  Your laptop / host                     SDK container (Docker)                    Modalix DevKit
  ──────────────────                     ──────────────────────                    ──────────────
  browser  ──https:9900──►  Neat Insight  (viewer + RTSP test streams)
                                         Model Compiler   (ONNX → .tar.gz)
                                         C++ cross-compiler (cmake)
                                         dk <binary> ───────── runs it on ──────►  ./app  (A65 CPU, EV74, MLA)
                                              ▲                                         │
                                              └──── /workspace is shared (NFS) ─────────┘
                                         Insight ◄── video UDP 9000+ch / metadata UDP 9100+ch ──┘
```

- **You edit and build** in the SDK container, inside `/workspace`.
- **`dk`** runs the ARM64 binary on the DevKit. The board sees the same `/workspace` path, so nothing is copied by hand.
- **Neat Insight** runs in the SDK container. It serves the RTSP test streams the apps consume, and it
  displays the video and metadata the apps send back.

---

## Session 0 — Installation checklist (45 min)

Do as much of this as possible **before** the workshop. The model compiler download alone is about 10 GB.

### Host requirements

| | |
|---|---|
| OS | Ubuntu 22.04 / 24.04 (Windows 11 via WSL2 and macOS on Apple Silicon also work) |
| Hardware | 4+ CPU cores, **16 GB RAM**, 100 GB free disk |
| Software | Docker, `sudo`, `git`, `curl` |
| Network | Host and DevKit on the same network: `ping <devkit-ip>` must work |
| Account | A SiMa developer account (for `sima-cli login`) |

### 1. Install `sima-cli` on the host

```bash
curl -fsSL https://artifacts.neat.sima.ai/sima-cli/linux-mac.sh | bash
source ~/.bashrc
sima-cli --version
sima-cli login
```

### 2. Check the DevKit software version

```bash
ssh sima@<devkit-ip>          # password: edgeai
cat /etc/buildinfo            # expect DISTRO_VERSION = 2.1.3
```

The board image, the SDK and the Neat Library are released as a set. Keep them on matching versions.

### 3. Install the Neat SDK (container + Insight + Neat Core on the board)

```bash
sima-cli neat install sdk@release-2.1
```

Answers to the setup prompts:

| Prompt | Answer |
|---|---|
| Pair this SDK with a DevKit now? | `y`, then enter the DevKit IP |
| Host workspace path | `~/workspace` |
| Create a new SDK container? | **first install:** yes. **Every later run:** `n` |
| Install the Model Compiler extension? | **`y`** (needed for session 2) |
| DevKit workspace path | `Enter` for `/workspace` |

To restart the SDK later (for example after a reboot or a DevKit IP change), run this on the host and answer `n` at the "new container" prompt:

```bash
sima-cli sdk setup --devkit <devkit-ip>
```

### 4. Attach VS Code to the container

Install the **Dev Containers** extension, then run `Ctrl+Shift+P` → **Dev Containers: Attach to Running Container…** → `sima-neat/sdk`.
Open the folder `/workspace`. From here on, every command runs in the **VS Code terminal inside the container**.

### 5. Verify the setup

```bash
dk status                    # SSH status: reachable, Sync method: nfs
neat                         # versions: core 0.5.0, Insight running
```

Open Insight in a browser at **`https://<host-ip>:9900`** (HTTPS; accept the certificate warning).

### 6. Add `ultralytics` to the Model Compiler environment

The model-compiler virtual environment does not include `ultralytics`, which session 2 uses to
export the models:

```bash
source /sdk-extensions/model-compiler/bin/activate
pip install "ultralytics==8.4.175"     # the version this workshop was tested with
python -c "import afe, ultralytics; print('model compiler OK')"
deactivate
```

If `import afe` fails with `libLLVM.so.18.1: cannot open shared object file`, install the missing
system libraries and try again: `sudo apt-get install -y libllvm18 libopenblas0`.

### 7. Start the test video streams in Insight

1. **Media Sources → Import Media → Catalog**: import a traffic video (for example
   `parking_garage_cars` or `sj_highway`) and a people video (for example `person_gym_jumping`).
   Pick the **720p30** H.264 variants (1280×720, 30 fps): the app configs expect 30 fps.
2. **Streaming**: assign the traffic video to `src1` and the people video to `src2`, then start both.

| Source | Content | Used by |
|---|---|---|
| `rtsp://<host-ip>:8554/src1` | cars / street | App 1, App 2 stream 0 (detection) |
| `rtsp://<host-ip>:8554/src2` | people | App 2 stream 1 (pose) |

The **Video Viewer** tab shows what the apps send back. Channel N there = video UDP port 9000+N.

You can check what is playing from the SDK terminal:

```bash
curl -sk https://127.0.0.1:9900/api/mediasrc | python3 -c 'import json,sys;[print(s["index"],s["state"],s["file"]) for s in json.load(sys.stdin) if s["file"]]'
```

> Use the **host IP** in RTSP URLs (`rtsp://<host-ip>:8554/...`), not `127.0.0.1`. The app runs on the
> DevKit, and `127.0.0.1` there means the DevKit itself. `curl -sk https://127.0.0.1:9900/api/server-ip`
> prints the right IP.

---

## Workshop folder layout

All material lives in one folder. Copy it to `/workspace/workshop` in the SDK container:

```text
workshop/
├── 00-agenda.md                  ← this file
├── 01-neat-cpp-api.md            ← session 1
├── 02-model-compilation.md       ← session 2
├── 03-app1-single-stream.md      ← session 3
├── 04-app2-dual-stream.md        ← session 4
├── model-info/                   ← session 1 hands-on: inspect + benchmark a model
├── model-compilation/            ← session 2: export_onnx.py, surgery.py, compile.py
├── app1-single-stream/           ← session 3: main.cpp, CMakeLists.txt, app1.conf
├── app2-dual-stream/             ← session 4: main.cpp, CMakeLists.txt, app2.conf
└── prebuilt/                     ← compiled archives from the instructor, not in the repo (fallback)
```

Every source file is also printed in full in its markdown file, so attendees can either use the
folder or create the files themselves by copying from the page.

### Prebuilt fallback

If a compile fails or runs long, hand out the archives in `prebuilt/`. They were built with exactly
the session 2 flow and validated with both apps on the DevKit:

| File | Precision | Use |
|---|---|---|
| `yolo26n.compile_ready_mpk.tar.gz` | INT8 | App 1, App 2 stream 0 |
| `yolo26n-pose.compile_ready_mpk.tar.gz` | BF16 | App 2 stream 1 |
| `yolo26n-bf16.compile_ready_mpk.tar.gz` | BF16 | Optional: compare with INT8 in App 1 |

```bash
cd /workspace/workshop
mkdir -p app1-single-stream/models app2-dual-stream/models
cp prebuilt/yolo26n.compile_ready_mpk.tar.gz                                  app1-single-stream/models/
cp prebuilt/yolo26n.compile_ready_mpk.tar.gz prebuilt/yolo26n-pose.compile_ready_mpk.tar.gz app2-dual-stream/models/
```

---

## Useful references

- Neat developer portal: <https://developer.sima.ai>
- Neat Apps examples: <https://github.com/sima-neat/apps/tree/main/examples>
- Neat Library headers inside the SDK: `/opt/toolchain/aarch64/modalix/usr/include` (start at `neat.h`)
