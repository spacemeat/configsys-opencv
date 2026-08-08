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
- **`gpu`** — GPU backends to compile. Tokens / aliases → CMake flags + required SDK:

  | token | alias | CMake | requires (SDK) |
  | --- | --- | --- | --- |
  | `cuda` | `nvidia` | `-DWITH_CUDA=ON -DOPENCV_DNN_CUDA=ON` | `cuda-toolkit` |
  | `hip` | `amd`, `rocm` | `-DWITH_HIP=ON` (experimental) | `rocm-hip` |

  Absent = CPU-only. Name the SDK in the **same binding's `requires:`** — the driver checks the
  toolchain is on `PATH` and refuses to build (loudly, before a 30-min compile) if it's missing,
  rather than silently falling back to CPU.
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
