# configsys-opencv — a from-source, GPU-capable OpenCV build

A [configsys](https://github.com/spacemeat/configsys) **code plugin** that adds build-from-source
methods to the base `opencv` component, with optional **CUDA** (or experimental **HIP**) GPU support
and `opencv_contrib` for the CUDA modules. Base configsys ships `opencv` as a CPU-only *native*
package; this plugin **adds** a pinnable method per variant (CPU / CUDA-11 / CUDA-12 / HIP) — all
under `opencv`, with `native` staying the default. Loading the plugin never changes a resolution;
you pick a source build explicitly with a pin:

```console
$ configsys where opencv        # methods valid HERE, e.g. on nvidia + CUDA 12:
                                #   native (default), opencv-build (CPU), opencv-cuda12
$ configsys pin opencv opencv-cuda12
$ configsys install opencv      # CUDA-12 stack from source (cuDNN 9, opencv_contrib)
```

It follows the [configsys-blender](https://github.com/spacemeat/configsys-blender) /
configsys-kicad pattern: GPU-SDK builds are exactly the "gnarly" case the general `configsys-source`
plugin bars, so OpenCV+CUDA gets its own small orchestration driver.

## The shape

| File | Role |
| --- | --- |
| `plugin.hu` | manifest — `provides.drivers: [opencv-build]`, `code: opencv.py`, `data: opencv.hu` |
| `opencv.hu` | adds the per-variant methods to base's `opencv` (CPU / CUDA-11 / CUDA-12 / HIP) |
| `opencv.py` | the `opencv-build` driver + thin per-variant subclasses — maps `gpu:` to CMake flags, validates the toolchain |
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
### The variant methods (pin one)

Each variant is its own `via:` (a thin subclass of the build driver, presetting the backend), so you
choose by **pinning** — no uncommenting, no copying bindings:

| `via:` | preset | `requires:` (the stack) | `when:` (where it's offered) |
| --- | --- | --- | --- |
| `opencv-build` | CPU | cmake, cpp-toolchain, git | anywhere |
| `opencv-cuda11` | CUDA + cuDNN | + cuda-toolkit-11, **cuDNN 8**, **gcc-10** | `gpu:nvidia and cuda < 12` |
| `opencv-cuda12` | CUDA + cuDNN | + cuda-toolkit-12, **cuDNN 9** | `gpu:nvidia and cuda >= 12` |

The CUDA toolkits (`cuda-toolkit-11` / `cuda-toolkit-12`), `cuda-repo`, and cuDNN (`cudnn-8` /
`cudnn-9`) live in **base configsys** (`routes.hu`) — they're general-purpose, not OpenCV-specific.
Both toolkits `provides: cuda-toolkit` (a generic capability); `-12` is the default provider and
`-11` is opt-in, pulled only when a stack names it. This plugin adds only the OpenCV build variants
and the `cuda` version facet.
| `opencv-hip` | HIP | + rocm-hip | `gpu:amd` |

The GPU methods keep a `when:` on the hardware they need — the two detected **facets** (see the base
`docs/facets.md`):

- **`gpu`** (vendor) — `lspci` (from **pciutils**; falls back to `nvidia-smi`/`rocminfo` on minimal
  images). Override if it misfires: `facets: { gpu: nvidia }` or `CONFIGSYS_FACET_gpu=nvidia`.
- **`cuda`** (version, from `nvcc --version`) — routes you to the CUDA-11 vs CUDA-12 stack.

Because the methods carry `when:`, the **Components** screen only offers the ones valid on this
machine (on an NVIDIA + CUDA-12 box: `native`, `opencv-build`, `opencv-cuda12`). The **Profiles**
screen lists them all, marking the ones not available here — you may be authoring for another
machine. `native` apt `opencv` stays the default; nothing builds until you pin a source method.

**Bootstrapping a box that doesn't have CUDA yet:** with no `nvcc`, the `cuda` facet is absent, so
neither CUDA method is offered. **Declare your target** — `CONFIGSYS_FACET_cuda=12` in the env, or:

```
facets: { cuda: 12 }
```

Then `opencv-cuda12` becomes available; pin it and configsys installs `cuda-toolkit-12` + cuDNN 9.
(If a *different* CUDA is already installed, configsys surfaces the conflict rather than silently
down-rev'ing it.)

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
configsys pin opencv opencv-cuda12                         # pick a variant (native is the default)
configsys install opencv
```

`get_version` reports installed **only for the variant actually built**: each build stamps a
`.configsys-variant` marker (its via + backends), and a hand-built tree is best-effort probed
(`cvconfig.h` HAVE_CUDA/HAVE_HIP + the linked cuDNN soname). So `configsys versions opencv` names the
built variant, and — even unpinned — `configsys inspect` surfaces it as *"also present"*. Built OpenCV
somewhere nonstandard? `locations: { opencv: /path/to/opencv-git }` in your config points at it.

## Status

**UNVERIFIED** — the recipe is transcribed from OpenCV's CMake build docs; a full CUDA build is long
(30+ min) and needs a matching `cuda-toolkit` with `nvcc` on `PATH`. The driver's orchestration
(gpu-field → flags, toolchain validation, state) is unit-tested; the compile itself is exercised by
hand. HIP/ROCm support in OpenCV is experimental.

## Tests

```sh
PYTHONPATH=/path/to/configsys python -m pytest test/ -q
```
