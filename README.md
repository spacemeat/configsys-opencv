# configsys-opencv — a from-source, GPU-capable OpenCV build

A [configsys](https://github.com/spacemeat/configsys) **code plugin** that adds a build-from-source
method to the base `opencv` component, with optional **CUDA** (or experimental **HIP**) GPU support
and `opencv_contrib` for the CUDA modules. Base configsys ships `opencv` as a CPU-only *native*
package; this plugin **adds** a `via: opencv-build` binding — `native` stays the default, so loading
the plugin never changes a resolution. You pick the GPU build explicitly.

It follows the [configsys-blender](https://github.com/spacemeat/configsys-blender) /
configsys-kicad pattern: GPU-SDK builds are exactly the "gnarly" case the general `configsys-source`
plugin bars, so OpenCV+CUDA gets its own small orchestration driver.

## The shape

| File | Role |
| --- | --- |
| `plugin.hu` | manifest — `provides.drivers: [opencv-build]`, `code: opencv.py`, `data: opencv.hu` |
| `opencv.hu` | adds the `via: opencv-build` binding to base's `opencv` (CPU default + GPU examples) |
| `opencv.py` | the `opencv-build` driver — maps `gpu:` to CMake flags, validates the toolchain |
| `build-opencv.sh` | the recipe: clone opencv (+ opencv_contrib), Release CMake build, install |
| `test/` | the driver's pure-logic unit tests |

## Binding fields

```
opencv: { install: [ { via: opencv-build  ref: 4.11.0  dir: opencv-git
                       gpu: [ cuda ]  requires: [ cmake, cpp-toolchain, cuda-toolkit ] } ] }
```

- **`ref`** — git tag/branch to build (empty = default branch).
- **`dir`** — build-tree parent (scope-honored); holds `opencv/` (+ `opencv_contrib/`).
- **`prefix`** — install prefix (default `~/.local`). Libraries land in `$PREFIX/lib` — add it to
  your loader path (`LD_LIBRARY_PATH` / `ldconfig`) and `$PREFIX/lib/pkgconfig` to `PKG_CONFIG_PATH`.
- **`gpu`** — GPU backends to compile. Tokens COMPOSE; name each token's SDK in the same binding's
  `requires:`.

  | token | CMake it adds | requires (SDK) | notes |
  | --- | --- | --- | --- |
  | `cuda` | `-DWITH_CUDA=ON` (+ `NVCUVID/NVCUVENC=OFF`) | `cuda-toolkit` | CUDA accel; **no** cuDNN needed |
  | `cudnn` | `-DOPENCV_DNN_CUDA=ON` | `cudnn` | the DNN CUDA backend; **implies `cuda`** |
  | `hip` | `-DWITH_HIP=ON` (experimental) | `rocm-hip` | AMD |

  Aliases: **`nvidia`** = `[cuda, cudnn]` (the full stack), `amd`/`rocm` = `[hip]`. Absent = CPU-only.

  The DNN CUDA backend (`cudnn` / `OPENCV_DNN_CUDA`) hard-requires cuDNN, an **extra** dependency on
  top of the CUDA toolkit — hence the separate `cudnn` token + component. Plain `gpu: [cuda]` builds
  fine with just the toolkit; add `cudnn` (or use `nvidia`) for the DNN backend. The hardware
  video-codec bits (`WITH_NVCUVID/NVCUVENC`) need NVIDIA's separate Video Codec SDK, so they're OFF
  by default (they otherwise spam "requires Video Codec SDK" warnings).

  The driver checks each backend's toolchain is present and refuses to build (loudly, before a
  30-min compile) if it's missing — no silent CPU fallback. For CUDA it also **auto-steers `nvcc`
  at a supported host gcc** (`-DCUDA_HOST_COMPILER=/usr/bin/gcc-N`) when the default `gcc` is newer
  than the CUDA header's `__GNUC__ > N` ceiling, and clears a stale `CMakeCache.txt` on a GPU
  reconfigure so CUDA is re-detected fresh.

  **cuDNN installs itself.** cuDNN isn't in the distro repos (NVIDIA license), so on debian/redhat
  the `cudnn` component `requires:` a **`cuda-repo`** helper that sets up NVIDIA's CUDA network repo
  first (Debian: the `cuda-keyring` `.deb`; RHEL: `dnf config-manager`), then installs `libcudnn8-dev`
  — no manual repo step. `cuda-repo` derives NVIDIA's distro-specific path (`ubuntu2204`/`debian12`/
  `rhel9`/…) from `/etc/os-release` (via `UBUNTU_CODENAME`, so pop/mint/elementary map to the right
  ubuntu segment). On **Arch**, cuDNN is in the official repos (`pacman cudnn`) so no helper is used.
  `libcudnn8-dev` is cuDNN-8 (CUDA 11 / early 12); for a cuDNN-9 / CUDA-12.3+ box swap the name to
  `libcudnn9-dev-cuda-12` in `opencv.hu`.
- **`contrib`** — clone `opencv_contrib`'s extra modules (bool). Default: ON when `gpu` is set (the
  CUDA modules — `cudaarithm`, `cudaimgproc`, … — live there), OFF for a plain CPU build.

## Use it

```sh
configsys plugin add github:spacemeat/configsys-opencv    # or a local path / file: source
configsys plugin trust configsys-opencv                   # it runs code, so trust is required
configsys pin set opencv opencv-build                      # opt into the source build (native is default)
# edit the opencv binding in your config to add `gpu: [ cuda ]  requires: [ ..., cuda-toolkit ]`
configsys install opencv
```

## Status

**UNVERIFIED** — the recipe is transcribed from OpenCV's CMake build docs; a full CUDA build is long
(30+ min) and needs a matching `cuda-toolkit` with `nvcc` on `PATH`. The driver's orchestration
(gpu-field → flags, toolchain validation, state) is unit-tested; the compile itself is exercised by
hand. HIP/ROCm support in OpenCV is experimental.

## Tests

```sh
PYTHONPATH=/path/to/configsys python -m pytest test/ -q
```
