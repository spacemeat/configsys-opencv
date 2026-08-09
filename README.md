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
### Automatic variant selection (facets)

You don't uncomment or copy a GPU binding — the right one is **auto-selected for this machine**.
`opencv` ships four real `opencv-build` bindings (CPU / CUDA-11 / CUDA-12 / HIP), gated on two
detected **facets** (see the base `docs/facets.md`):

- **`gpu`** (vendor) → `gpu:nvidia` picks a CUDA build, `gpu:amd` picks HIP, neither picks the CPU
  source build. Detected via `lspci` (from **pciutils** — present on ~every workstation, but not
  minimal images/containers); if `lspci` is absent it falls back to `nvidia-smi`/`rocminfo`. If
  detection ever misfires, override it: `facets: { gpu: nvidia }` in the config, or
  `CONFIGSYS_FACET_gpu=nvidia`.
- **`cuda`** (version, from `nvcc --version`) → `cuda < 12` picks the **CUDA-11 stack** (cuDNN 8 +
  gcc-10), `cuda >= 12` picks the **CUDA-12 stack** (cuDNN 9). Each stack has **explicit, versioned
  dependencies** — no guessing, no install-time surprises.

Native apt `opencv` stays the default; the source build is opt-in
(`configsys pin set opencv opencv-build`), and *then* hardware picks the variant. So on your NVIDIA
+ CUDA-11 box, pinning the source build gives the CUDA-11 stack automatically.

**Bootstrapping a box that doesn't have CUDA yet:** the `cuda` facet is absent, so no CUDA variant
matches and you'd get the CPU build. **Declare your target** instead of guessing — either
`CONFIGSYS_FACET_cuda=12` in the env, or in your config:

```
facets: { cuda: 12 }
```

Then the CUDA-12 stack resolves and configsys installs `cuda-toolkit-12` + cuDNN 9. (If a *different*
CUDA is already installed, configsys surfaces the conflict rather than silently down-rev'ing it.)

`when:` clauses only ever test **stable facts** (OS, `gpu`, `cuda`) — version-coupling lives in
`requires:`, so there's one consistent gating model.

- **`gpu`** (binding field) — GPU backends the driver compiles. Tokens COMPOSE; name each token's SDK
  in the same binding's `requires:`.

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
  at a workable host gcc** (`-DCUDA_HOST_COMPILER=/usr/bin/gcc-N`): it reads nvcc's `__GNUC__ > N`
  ceiling from `crt/host_config.h` and, if the default `gcc` is newer, points nvcc at the newest
  installed gcc within the ceiling. **Special case:** when the ceiling is 11 (the CUDA 11.4–11.6
  era), gcc 11 itself trips a known nvcc bug parsing gcc-11's `<functional>`/`std_function.h`
  (*"parameter packs not expanded"*), so the driver steps the target down to **gcc 10** (the
  known-good host for that CUDA line). If no usable gcc is installed it **fails fast with the fix**
  (`apt install gcc-10 g++-10`) instead of dying mid-compile. Override with **`cuda-host-compiler:
  /usr/bin/gcc-N`** on the binding. It also clears a stale `CMakeCache.txt` on a GPU reconfigure so
  CUDA is re-detected fresh.

  **cuDNN installs itself** (repo + package + loader cache). Not in the distro repos (NVIDIA
  license), so on debian/redhat the `cudnn` component `requires:` a **`cuda-repo`** helper that sets
  up NVIDIA's CUDA network repo (Debian: the `cuda-keyring` `.deb`; RHEL: `dnf config-manager`),
  derived from `/etc/os-release` (`ubuntu2204`/`debian12`/`rhel9`/… — via `UBUNTU_CODENAME` so
  pop/mint/elementary map to the right ubuntu segment). It then installs `libcudnn8-dev` **and runs
  `ldconfig`** — NVIDIA's libcudnn8 packages skip the ldconfig trigger, so without this the runtime
  `.so` isn't in the loader cache and the built OpenCV can't load it. On **Arch**, cuDNN is in the
  official repos (`pacman cudnn`, ldconfig handled) so no helper is used. `libcudnn8-dev` is cuDNN-8
  (CUDA 11 / early 12); for a cuDNN-9 / CUDA-12.3+ box swap the name to `libcudnn9-dev-cuda-12`.
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
