# Workshop validation (instructor only)

Tools that keep the workshop markdown and the tested code identical, and prove that the code runs.

| File | Purpose |
|---|---|
| `docs-src/*.md` | Markdown sources. `{{include:<path>}}` markers pull in the real source files |
| `build_docs.py` | Expands the includes and writes `../01-…04-*.md`. Run it after **any** code change |
| `validate_md.sh` | Extracts every code block from the generated `.md` files into `from-md/`, diffs each against the sources, cross-builds all three C++ projects, and runs them on the paired DevKit |
| `pose_check.png` | The BF16 pose model's keypoints drawn on a decoded test frame |

```bash
cd /workspace/workshop/_validation
python3 build_docs.py                                   # regenerate the markdown
bash validate_md.sh <host-ip> src1 src2                 # <det RTSP slot> <pose RTSP slot>
```

Every check must pass before the markdown is published.
