'''opencv.py — the `opencv-build` driver for the configsys-opencv plugin.

Builds OpenCV from source with optional GPU (CUDA / experimental HIP) support. The driver does the
safe, generic orchestration (locate + run the recipe, translate the `gpu:` field into CMake flags,
validate the toolchain, report state); the actual recipe lives in `build-opencv.sh` next to this
file — edit that to own the tweakable bits (extra cmake flags, the OpenCV version). Mirrors the
configsys-blender / configsys-kicad pattern.

Component shape (added to base's `opencv`, alongside its native CPU binding):
    opencv: { install: [ { via: opencv-build  ref: 4.11.0  dir: opencv-git
                           gpu: [ cuda ]  requires: [ cmake, cpp-toolchain, cuda-toolkit ] } ] }
  ref      git tag/branch to build   (empty = default branch)
  dir      build-tree parent, scope-honored (bare-relative -> under ~ ; contains opencv/ [+ opencv_contrib/])
  prefix   install prefix ($PREFIX); default ~/.local. Libs land in $PREFIX/lib — put it on your
           loader path (LD_LIBRARY_PATH / ldconfig) and $PREFIX/lib/pkgconfig on PKG_CONFIG_PATH.
  gpu      GPU backends to compile: a list of tokens (cuda, hip) or vendor aliases (nvidia -> cuda,
           amd/rocm -> hip). Absent = CPU-only. The driver maps these to WITH_CUDA/WITH_HIP CMake
           flags AND validates each backend's toolchain is present (declared via the same binding's
           `requires:`, e.g. cuda-toolkit). A gpu build auto-clones opencv_contrib (the cuda modules
           live there). A mismatch (gpu names a backend whose toolchain isn't installed) is caught
           here, loudly, before a long build — never a silent CPU fallback.
  contrib  clone opencv_contrib's extra modules (bool). Default: ON when gpu is set (cuda needs it),
           OFF for a plain CPU build.

`get_version` reports built once a libopencv_core*.so exists in the build tree. Uninstall LEAVES the
source tree in place (auto-removing a checkout with local work is too destructive). User-space.
'''

import shlex
from pathlib import Path

from configsys.plugins import Driver, Result

# GPU backend token -> the CMake -D flags it turns on. Tokens are GRANULAR and compose:
#   cuda   WITH_CUDA (core/imgproc/cudaarithm/... acceleration); needs only the CUDA toolkit.
#   cudnn  the DNN CUDA backend (OPENCV_DNN_CUDA); needs cuDNN ON TOP of cuda -> IMPLIES cuda
#          (auto-added in _gpu_backends) and its OWN `cudnn` dependency component + probe.
#   hip    AMD ROCm/HIP (experimental in OpenCV 4.x).
# The hardware video-codec bits (WITH_NVCUVID/NVCUVENC) need NVIDIA's separate Video Codec SDK, so
# they're OFF by default (a from-source build otherwise emits noisy "requires Video Codec SDK"
# warnings) — opt back in per-binding with an extra `-D...=ON` in build-opencv.sh if you have it.
_GPU_FLAGS = {
    'cuda':  ['-DWITH_CUDA=ON', '-DWITH_NVCUVID=OFF', '-DWITH_NVCUVENC=OFF'],
    'cudnn': ['-DOPENCV_DNN_CUDA=ON'],                 # rides on cuda's WITH_CUDA
    'hip':   ['-DWITH_HIP=ON'],
}
# each backend -> the toolchain checks it needs, as (probe cmd, SDK component, human label). The SDK
# name is what the binding's `requires:` should list AND what the "missing" error points at. A
# backend may need MORE than one (cudnn wants the cuDNN headers/libs specifically).
_GPU_PROBE = {
    'cuda':  [('command -v nvcc', 'cuda-toolkit', 'nvcc (CUDA toolkit)')],
    # `ls A B C 2>/dev/null | grep -q .` — prints the paths that DO exist (errors for the misses go
    # to /dev/null) and succeeds if ANY line came out. NOT `ls A B C >/dev/null 2>&1`: that exits
    # non-zero whenever *any* listed path is missing, even when cudnn.h is present. ldconfig is a
    # secondary check (the .so may be installed before its cache is refreshed, so header-first).
    'cudnn': [('ls /usr/include/cudnn.h /usr/include/*/cudnn.h /usr/local/cuda*/include/cudnn.h '
               '2>/dev/null | grep -q . || ldconfig -p 2>/dev/null | grep -q libcudnn',
               'cudnn', 'cuDNN headers/libs')],
    'hip':   [('command -v hipcc', 'rocm-hip', 'hipcc (ROCm)')],
}
# vendor aliases. `nvidia` is the FULL NVIDIA stack (CUDA + the DNN cuDNN backend) — the common "give
# me everything" choice; `cuda`/`cudnn` stay available for a leaner or à-la-carte build.
_GPU_ALIAS = {'nvidia': ['cuda', 'cudnn'], 'amd': ['hip'], 'rocm': ['hip']}


class OpencvBuild(Driver):
    name = 'opencv-build'
    privileged = False
    default_scope = 'user'
    honors_scope = False        # single build tree; rebuild for a different scope by hand

    def _build_dir(self, rc):
        return self.scoped_dir(rc.fields.get('dir') or 'opencv-git', rc)

    def _prefix(self, rc):
        p = rc.fields.get('prefix')
        if p:
            return self.scoped_dir(p, rc)
        base = self.paths.home if self.paths is not None else Path.home()
        return (base / '.local') if self._scope(rc) == 'user' else Path('/usr/local')

    def _script(self, rc):
        # build-opencv.sh ships in THIS driver's plugin dir (next to opencv.py) — find it via
        # __file__, NOT next to rc.source: the binding may be overridden in a user's layer (which is
        # the normal way to set gpu:/ref:) that doesn't carry the recipe.
        return Path(__file__).resolve().parent / 'build-opencv.sh'

    # -- gpu backends -----------------------------------------------------

    def _gpu_backends(self, rc):
        '''The `gpu:` field expanded to canonical tokens (aliases resolved, deduped, order kept).
        Raises ValueError on an unknown token.'''
        raw = rc.fields.get('gpu') or []
        if isinstance(raw, str):
            raw = [raw]
        out = []
        for tok in raw:
            for b in _GPU_ALIAS.get(tok, [tok]):
                if b not in _GPU_FLAGS:
                    raise ValueError(f'unknown gpu backend {b!r} (want one of {", ".join(_GPU_FLAGS)}, '
                                     f'or an alias {", ".join(_GPU_ALIAS)})')
                if b not in out:
                    out.append(b)
        # the DNN cuDNN backend rides on WITH_CUDA — asking for cudnn implies cuda (so its flags,
        # probe, and requires: all come along); insert cuda just before cudnn to keep order sane.
        if 'cudnn' in out and 'cuda' not in out:
            out.insert(out.index('cudnn'), 'cuda')
        return out

    def _cuda_host_compiler_flag(self, rc):
        '''Steer nvcc at a host gcc it can actually use. Returns (flag, error): `flag` is a
        `-DCUDA_HOST_COMPILER=...` string ('' if the default gcc is fine), `error` a message when no
        usable gcc is installed (so install() can fail fast with a fix, not 20 min into a cryptic
        compile). Priority: an explicit binding `cuda-host-compiler:` wins; else auto-detect.

        Auto-detect ceiling = nvcc's `#if __GNUC__ > N` in crt/host_config.h (too-new a host gcc dies
        with "unsupported GNU version"). SPECIAL CASE: when the ceiling is 11 — the CUDA 11.4–11.6 era
        — gcc 11 itself trips a well-known nvcc bug parsing gcc-11's <functional>/std_function.h
        ("parameter packs not expanded"), so we step the target down to gcc 10 (the known-good host
        for that CUDA line; still within the ceiling). Read-only + in-process, honest under --pretend.'''
        import re
        import shutil
        import subprocess
        override = rc.fields.get('cuda-host-compiler')
        if override:
            return f'-DCUDA_HOST_COMPILER={override}', None
        nvcc = shutil.which('nvcc')
        if not nvcc:
            return '', None
        ceiling = None
        for h in (Path(nvcc).resolve().parent.parent / 'include' / 'crt' / 'host_config.h',
                  Path('/usr/include/crt/host_config.h')):
            try:
                m = re.search(r'__GNUC__\s*>\s*(\d+)', h.read_text())
            except OSError:
                continue
            if m:
                ceiling = int(m.group(1))
                break
        if ceiling is None:
            return '', None
        if ceiling == 11:
            ceiling = 10               # dodge the gcc-11 std_function.h nvcc bug (CUDA 11.4–11.6)
        try:
            major = int(subprocess.run(['gcc', '-dumpversion'], capture_output=True,
                                       text=True).stdout.strip().split('.')[0])
        except (ValueError, OSError):
            return '', None
        if major <= ceiling:
            return '', None                            # the default gcc is already acceptable
        for k in range(ceiling, 6, -1):                # newest installed gcc-K that nvcc can use
            alt = Path(f'/usr/bin/gcc-{k}')
            if alt.exists():
                return f'-DCUDA_HOST_COMPILER={alt}', None
        return None, (f"CUDA needs a host gcc <= {ceiling} (default gcc is {major}), and no "
                      f"/usr/bin/gcc-{ceiling} is installed — run `sudo apt install gcc-{ceiling} "
                      f"g++-{ceiling}`, or set `cuda-host-compiler:` on the binding")

    def _gpu_cmake(self, backends):
        '''The deduped CMake flag string for a set of backends (empty = CPU-only).'''
        flags = []
        for b in backends:
            for f in _GPU_FLAGS[b]:
                if f not in flags:
                    flags.append(f)
        return ' '.join(flags)

    def _contrib(self, rc, backends):
        '''Whether to clone opencv_contrib (its cuda modules are needed for a GPU build). Defaults ON
        when gpu is set, OFF for a plain CPU build; a binding may force it with `contrib: true`.'''
        v = rc.fields.get('contrib')
        if v is None:
            return bool(backends)
        return str(v).lower() in ('true', '1', 'yes', 'on')

    # -- read -------------------------------------------------------------

    def get_version(self, rc):
        '''"installed" = a built libopencv_core*.so exists in the build tree. The version is what the
        source is checked out at (`git describe --tags`) — for a tag build that equals get_latest.'''
        root = self._build_dir(rc)
        lib = root / 'opencv' / 'build' / 'lib'
        r = self.runner.run(f'ls {shlex.quote(str(lib))}/libopencv_core.so* 2>/dev/null')
        if not (r and r.ok and r.stdout.strip()):
            return None
        src = root / 'opencv'
        d = self.runner.run(f'git -C {shlex.quote(str(src))} describe --tags')
        return (d.stdout.strip() if d.ok else '') or 'built'

    def get_latest(self, rc):
        return rc.fields.get('ref') or 'built'

    def is_locked(self, rc):
        return False

    # -- mutate -----------------------------------------------------------

    def install(self, rc):
        script = self._script(rc)
        if not script.exists():
            return Result(f'(opencv-build: recipe {script} not found)', 1)
        try:
            backends = self._gpu_backends(rc)
        except ValueError as e:
            return Result(f'(opencv-build: {e})', 1)
        # each requested backend needs its toolchain on PATH — declared via the same binding's
        # `requires:` so resolution installs it. Verify here; fail loud rather than a silent CPU
        # build. (Under --pretend every probe reports ok, so a dry run never spuriously blocks.)
        for b in backends:
            for probe, sdk, label in _GPU_PROBE[b]:
                if not self.runner.run(probe).ok:
                    return Result(f"(opencv-build: gpu {b!r} needs {label}, which isn't present — add "
                                  f"the {sdk!r} component to this binding's requires:, then sync)", 1)
        gpu_cmake = self._gpu_cmake(backends)
        if any(b in ('cuda', 'cudnn') for b in backends):
            # steer nvcc at a gcc it supports (default too new / gcc-11 std_function bug); fail fast
            # with a fix if no usable host gcc is installed, rather than 20 min into a cryptic compile
            hc, err = self._cuda_host_compiler_flag(rc)
            if err:
                return Result(f'(opencv-build: {err})', 1)
            if hc:
                gpu_cmake = f'{gpu_cmake} {hc}'.strip()
        contrib = '1' if self._contrib(rc, backends) else '0'
        ref = shlex.quote(rc.fields.get('ref') or '')
        d = shlex.quote(str(self._build_dir(rc)))
        prefix = shlex.quote(str(self._prefix(rc)))
        env = f'GPU_CMAKE={shlex.quote(gpu_cmake)} ' if gpu_cmake else ''
        return self.runner.run(
            f'{env}bash {shlex.quote(str(script))} {ref} {d} {prefix} {contrib}', capture=False)

    def upgrade(self, rc):
        return self.install(rc)   # fetch + checkout + rebuild

    def set_version(self, rc, version):
        return self.install(rc)

    def uninstall(self, rc):
        return Result(f'(opencv-build: leaving {self._build_dir(rc)} in place; remove it by hand)', 0)

    def lock(self, rc):
        return Result('(opencv-build lock recorded in ledger)', 0)

    def unlock(self, rc):
        return Result('(opencv-build unlock recorded in ledger)', 0)

    def location(self, rc):
        return str(self._build_dir(rc))


DRIVERS = [OpencvBuild]
