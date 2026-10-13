#!/usr/bin/env bash
# Validate the workshop markdown the way an attendee uses it:
#   1. extract every "<!-- file: X -->" code block from the .md files into a FRESH folder
#   2. diff each extracted file against the tested source in the workshop folder
#   3. cross-build all three C++ projects from the extracted code
#   4. run them on the DevKit: model_info, App 1, App 2
#
# Usage (inside the SDK container):  bash validate_md.sh <host-ip> [rtsp-det-src] [rtsp-pose-src]
set -uo pipefail

HOST_IP=${1:?usage: validate_md.sh <host-ip> [det-src] [pose-src]}
DET_SRC=${2:-src1}
POSE_SRC=${3:-src2}
WS=/workspace/workshop
OUT=$WS/_validation/from-md
DEVKIT=$(bash -ic 'dk status' 2>/dev/null </dev/null | sed -n 's/.*sima@\([0-9.]*\):.*/\1/p')
[ -n "$DEVKIT" ] || { echo "no paired DevKit (dk status)"; exit 1; }
PASS=0; FAIL=0
ok()   { echo "  PASS  $*"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL+1)); }
on_board() { ssh -o BatchMode=yes sima@"$DEVKIT" "cd $1 && $2" 2>&1; }

echo "== 1. extract code blocks from markdown -> $OUT"
rm -rf "$OUT" && mkdir -p "$OUT"
python3 - "$WS" "$OUT" <<'EOF'
import re, sys, pathlib
ws, out = map(pathlib.Path, sys.argv[1:])
pat = re.compile(r"<!-- file: (\S+) -->\n```[a-z]*\n(.*?)\n```", re.S)
for md in sorted(ws.glob("0*.md")):
    for rel, body in pat.findall(md.read_text()):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(body + "\n")
        print(f"  {md.name:28s} -> {rel}")
EOF

echo "== 2. extracted files identical to tested sources"
for f in $(cd "$OUT" && find . -type f | sort); do
  if diff -q "$OUT/$f" "$WS/$f" >/dev/null; then ok "$f"; else bad "$f differs"; fi
done

echo "== 3. build from extracted code"
for p in model-info app1-single-stream app2-dual-stream; do
  if cmake -S "$OUT/$p" -B "$OUT/$p/build" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_PREFIX_PATH=/opt/toolchain/aarch64/modalix/usr >/dev/null 2>&1 \
     && cmake --build "$OUT/$p/build" -j 2>&1 | grep -q "Built target"; then ok "build $p"
  else bad "build $p"; fi
done
python3 -m py_compile "$OUT"/model-compilation/*.py && ok "python scripts compile" || bad "python scripts"

echo "== 4. run on the DevKit ($DEVKIT)"
# models produced by the session-2 flow
for p in app1-single-stream app2-dual-stream; do mkdir -p "$OUT/$p/models"; done
cp "$WS"/app1-single-stream/models/yolo26n.compile_ready_mpk.tar.gz "$OUT/app1-single-stream/models/"
cp "$WS"/app2-dual-stream/models/*.tar.gz "$OUT/app2-dual-stream/models/"

r=$(on_board "$OUT/model-info" "./build/model_info ../app1-single-stream/models/yolo26n.compile_ready_mpk.tar.gz")
echo "$r" | grep -q "latency_ms=" && ok "model_info: $(echo "$r" | grep latency_ms=)" || bad "model_info"

cd "$OUT/app1-single-stream"
sed -e "s/<host-ip>/$HOST_IP/g" -e "s#/src1#/$DET_SRC#" app1.conf > rtsp.conf
r=$(on_board "$PWD" "timeout -s INT 20 ./build/app1_single_stream rtsp.conf")
echo "$r" | grep -q "frames=200" && ok "app1 rtsp: $(echo "$r" | grep frames=200)" || { bad "app1 rtsp"; echo "$r" | tail -5; }

cd "$OUT/app2-dual-stream"
sed -e "s/<host-ip>/$HOST_IP/g" -e "s#/src1#/$DET_SRC#" -e "s#/src2#/$POSE_SRC#" app2.conf > rtsp.conf
r=$(on_board "$PWD" "timeout -s INT 20 ./build/app2_dual_stream rtsp.conf")
echo "$r" | grep -q "stream 0 det \] frames=200" && echo "$r" | grep -q "stream 1 pose\] frames=200" \
  && ok "app2 rtsp: both streams >= 200 frames" || { bad "app2 rtsp"; echo "$r" | tail -5; }

r=$(on_board "$OUT/app2-dual-stream" "./build/app2_dual_stream app2.conf")
echo "$r" | grep -q "edit the config first" && ok "placeholder guard" || bad "placeholder guard"

echo "== result: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
