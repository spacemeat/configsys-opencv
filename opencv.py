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

# GPU backend token -> the CMake -D flags it turns on, and -> (toolchain probe, SDK component). The
# SDK name is what the binding's `requires:` should list AND what the error message points at.
_GPU_FLAGS = {
    'cuda': ['-DWITH_CUDA=ON', '-DOPENCV_DNN_CUDA=ON'],
    'hip':  ['-DWITH_HIP=ON'],                          # experimental in OpenCV 4.x
}
_GPU_PROBE = {
    'cuda': ('command -v nvcc',  'cuda-toolkit'),
    'hip':  ('command -v hipcc', 'rocm-hip'),
}
_GPU_ALIAS = {'nvidia': ['cuda'], 'amd': ['hip'], 'rocm': ['hip']}


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
        return out

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
            probe, sdk = _GPU_PROBE[b]
            if not self.runner.run(probe).ok:
                return Result(f"(opencv-build: gpu {b!r} requested but its toolchain is missing — add "
                              f"the {sdk!r} component to this binding's requires:, then sync)", 1)
        gpu_cmake = self._gpu_cmake(backends)
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
