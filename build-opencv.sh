#!/usr/bin/env bash
# build-opencv.sh — the OpenCV from-source recipe for the `opencv-build` driver.
#
#   build-opencv.sh <ref> <dir> <prefix> <contrib:0|1>     (GPU_CMAKE env optional)
#
# Clones/updates opencv (and, when contrib=1, opencv_contrib) at <ref> under <dir>, then a Release
# CMake build into <dir>/opencv/build installed to <prefix>. GPU_CMAKE carries the -DWITH_CUDA/HIP
# flags the driver derived from the binding's gpu: field. Userland by default (no sudo); for a
# system <prefix> the final `cmake --install` may need write access — run configsys with a system
# scope or add sudo to the install line yourself.
#
# This is the WAVY part: pin the OpenCV version via the binding's `ref`, add extra -D flags here
# (e.g. -DWITH_TBB=ON, -DCPU_BASELINE=…), or point at your own contrib fork. UNVERIFIED — a real
# CUDA build is long (30+ min) and needs a matching cuda-toolkit + nvcc on PATH.
set -euo pipefail

REF="${1:-}"
DIR="${2:?build dir required}"
PREFIX="${3:?install prefix required}"
CONTRIB="${4:-0}"
GPU_CMAKE="${GPU_CMAKE:-}"

clone_at() { # <repo-url> <dest>
    local url="$1" dest="$2"
    if [ -d "$dest/.git" ]; then
        git -C "$dest" fetch --tags --force origin
    else
        git clone "$url" "$dest"
    fi
    if [ -n "$REF" ]; then
        git -C "$dest" checkout --quiet "$REF"
    fi
}

mkdir -p "$DIR"
clone_at "https://github.com/opencv/opencv" "$DIR/opencv"

EXTRA=""
if [ "$CONTRIB" = "1" ]; then
    clone_at "https://github.com/opencv/opencv_contrib" "$DIR/opencv_contrib"
    EXTRA="-DOPENCV_EXTRA_MODULES_PATH=$DIR/opencv_contrib/modules"
fi

# A GPU build caches the CUDA host compiler + detected arch + cuDNN result in CMakeCache.txt; a stale
# cache (e.g. from a first attempt under a too-new gcc) silently sticks across reconfigures. On a GPU
# (re)configure, drop just the cache so CUDA is re-detected fresh — object files under build/ stay.
# (The driver passes -DCUDA_HOST_COMPILER in GPU_CMAKE when the default gcc is too new for nvcc.)
if [ -n "$GPU_CMAKE" ] && [ -f "$DIR/opencv/build/CMakeCache.txt" ]; then
    echo "opencv-build: GPU build — clearing stale CMakeCache for a fresh CUDA detect"
    rm -f "$DIR/opencv/build/CMakeCache.txt"
fi

# shellcheck disable=SC2086  # GPU_CMAKE / EXTRA are intentional word-split flag lists
cmake -S "$DIR/opencv" -B "$DIR/opencv/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    $EXTRA $GPU_CMAKE

cmake --build "$DIR/opencv/build" -j"$(nproc)"
cmake --install "$DIR/opencv/build"

echo "opencv-build: installed to $PREFIX (libs in $PREFIX/lib — add to LD_LIBRARY_PATH/ldconfig)"
