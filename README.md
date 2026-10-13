# Building Edge AI Video Apps with SiMa Neat on Modalix

A hands-on workshop (about 2.5 hours) for building real-time video AI apps in C++ with the
SiMa Neat Library on a Modalix DevKit.

You will:

1. Compile two YOLO26 models yourself, one in **INT8** and one in **BF16**.
2. Build a **single-stream** app: one video source → one model → live video with boxes in Neat Insight.
3. Build a **dual-stream** app: two video sources → two models (detection and pose) → two live outputs.

**Tested with:** Modalix DevKit, board software 2.1.3, Neat SDK 2.1.3.1, Neat Library 0.5.0.

## Start here

Open **[00-agenda.md](00-agenda.md)**. It has the schedule, the installation checklist, and links to every session.

| # | Session | Guide | Code |
|---|---|---|---|
| 0 | Installation and introduction | [00-agenda.md](00-agenda.md) | |
| 1 | Neat C++ API introduction | [01-neat-cpp-api.md](01-neat-cpp-api.md) | [model-info/](model-info/) |
| 2 | Model compilation: INT8 and BF16 | [02-model-compilation.md](02-model-compilation.md) | [model-compilation/](model-compilation/) |
| 3 | App 1: one source, one model, one stream | [03-app1-single-stream.md](03-app1-single-stream.md) | [app1-single-stream/](app1-single-stream/) |
| 4 | App 2: two sources, two models, two streams | [04-app2-dual-stream.md](04-app2-dual-stream.md) | [app2-dual-stream/](app2-dual-stream/) |

## Getting the material

The guides expect the folder at `/workspace/workshop` inside the SDK container:

```bash
cd /workspace
git clone https://github.com/smaurya01/neat-workshop.git workshop
```

## Prebuilt models

If you can't compile the models yourself, ask the instructor for the prebuilt model archives
(`*.tar.gz`). They are not stored in this repository. Put them in a `prebuilt/` folder, then see
[Prebuilt fallback](00-agenda.md#prebuilt-fallback) for where to copy them.

## Repository layout

```text
├── 00-agenda.md … 04-app2-dual-stream.md   workshop guides
├── model-info/            session 1: inspect and benchmark a model
├── model-compilation/     session 2: export_onnx.py, surgery.py, compile.py
├── app1-single-stream/    session 3: main.cpp, CMakeLists.txt, app1.conf
└── app2-dual-stream/      session 4: main.cpp, CMakeLists.txt, app2.conf
```

Build output (`build/`), the model compilation working folder (`model-compilation/work/`) and model
archives (`*.tar.gz`) are not tracked. Your own builds and compiles create them. Get the prebuilt
archives from the instructor.

## References

- Neat developer portal: <https://developer.sima.ai>
- Neat Apps examples: <https://github.com/sima-neat/apps/tree/main/examples>
