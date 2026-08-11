'''Unit tests for the opencv-build driver's pure logic (gpu-field expansion, CMake flags, contrib
default). The build itself (build-opencv.sh) is exercised by hand — configsys must be importable
(run with the repo on PYTHONPATH).'''

import importlib.util
import pathlib

import pytest

_p = pathlib.Path(__file__).resolve().parent.parent / 'opencv.py'
_spec = importlib.util.spec_from_file_location('opencv', _p)
opencv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opencv)


class _RC:
    def __init__(self, **fields):
        self.fields = fields


def _drv():
    return opencv.OpencvBuild.__new__(opencv.OpencvBuild)   # no runner needed for the pure methods


def test_gpu_expansion_and_aliases():
    d = _drv()
    assert d._gpu_backends(_RC(gpu=['cuda'])) == ['cuda']
    assert d._gpu_backends(_RC(gpu=['nvidia'])) == ['cuda', 'cudnn']   # full NVIDIA stack
    assert d._gpu_backends(_RC(gpu=['amd'])) == ['hip']
    assert d._gpu_backends(_RC(gpu=['rocm'])) == ['hip']
    assert d._gpu_backends(_RC(gpu='cuda')) == ['cuda']               # scalar accepted
    assert d._gpu_backends(_RC(gpu=['cuda', 'cuda'])) == ['cuda']     # deduped
    assert d._gpu_backends(_RC()) == []                              # absent -> CPU-only
    with pytest.raises(ValueError):
        d._gpu_backends(_RC(gpu=['nonsense']))


def test_cudnn_implies_cuda():
    d = _drv()
    assert d._gpu_backends(_RC(gpu=['cudnn'])) == ['cuda', 'cudnn']         # cuda auto-added
    assert d._gpu_backends(_RC(gpu=['cuda', 'cudnn'])) == ['cuda', 'cudnn']  # order kept, no dup


def test_gpu_cmake_flags():
    d = _drv()
    assert d._gpu_cmake([]) == ''                                    # CPU -> no flags
    cuda = d._gpu_cmake(['cuda'])
    assert '-DWITH_CUDA=ON' in cuda
    assert '-DOPENCV_DNN_CUDA=ON' not in cuda                        # plain cuda: no DNN backend
    assert '-DWITH_NVCUVID=OFF' in cuda and '-DWITH_NVCUVENC=OFF' in cuda  # codec SDK off (no warnings)
    full = d._gpu_cmake(['cuda', 'cudnn'])
    assert '-DWITH_CUDA=ON' in full and '-DOPENCV_DNN_CUDA=ON' in full
    assert d._gpu_cmake(['hip']) == '-DWITH_HIP=ON'


def test_contrib_default_follows_gpu_unless_forced():
    d = _drv()
    assert d._contrib(_RC(), []) is False                          # plain CPU -> no contrib
    assert d._contrib(_RC(), ['cuda']) is True                     # gpu -> contrib (cuda modules)
    assert d._contrib(_RC(contrib='true'), []) is True             # forced on for a CPU build
    assert d._contrib(_RC(contrib='false'), ['cuda']) is False     # forced off


def test_cudnn_probe_survives_missing_paths():
    # regression: `ls A B C >/dev/null 2>&1` exits non-zero if ANY listed path is missing — even when
    # /usr/include/cudnn.h IS present — so it must use the `ls ... 2>/dev/null | grep -q .` form,
    # which prints the paths that exist and succeeds if any did.
    probe = opencv._GPU_PROBE['cudnn'][0][0]
    assert '2>/dev/null | grep -q .' in probe
    assert '>/dev/null 2>&1 ||' not in probe


def test_cuda_host_compiler_flag_no_nvcc(monkeypatch):
    # no nvcc on PATH -> no host-compiler flag, no error (never blocks / never guesses)
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    assert _drv()._cuda_host_compiler_flag(_RC()) == ('', None)


def test_cuda_host_compiler_override_wins():
    # an explicit binding field short-circuits auto-detection (needs no nvcc)
    rc = _RC(**{'cuda-host-compiler': '/usr/bin/gcc-10'})
    assert _drv()._cuda_host_compiler_flag(rc) == ('-DCUDA_HOST_COMPILER=/usr/bin/gcc-10', None)


def _variant(name):
    cls = next(d for d in opencv.DRIVERS if d.name == name)
    return cls.__new__(cls)


def test_driver_registration_shape():
    assert opencv.OpencvBuild.name == 'opencv-build'
    # one pinnable via per variant
    assert [d.name for d in opencv.DRIVERS] == [
        'opencv-build', 'opencv-cuda11', 'opencv-cuda12', 'opencv-hip']
    # each variant via presets its backend set (a binding `gpu:` still overrides)
    assert _variant('opencv-build')._gpu_backends(_RC()) == []
    assert _variant('opencv-cuda11')._gpu_backends(_RC()) == ['cuda', 'cudnn']
    assert _variant('opencv-cuda12')._gpu_backends(_RC()) == ['cuda', 'cudnn']
    assert _variant('opencv-hip')._gpu_backends(_RC()) == ['hip']
    assert _variant('opencv-cuda12')._gpu_backends(_RC(gpu=['hip'])) == ['hip']   # override


# -- integration: each stack variant is its own pinnable via ----------------------

def _base_routes():
    import configsys
    r = pathlib.Path(configsys.__file__).resolve().parent.parent / 'routes.hu'
    return str(r) if r.exists() else None


def test_variant_marker_discriminates_get_version():
    from configsys.runner import Result

    class Fake:
        '''A built opencv whose marker records `via=<marked>`.'''
        def __init__(self, marked):
            self.marked = marked

        def run(self, cmd, **kw):
            if cmd.startswith('ls ') and 'libopencv_core' in cmd:
                return Result(cmd, 0, stdout='libopencv_core.so.4.11\n')
            if cmd.startswith('cat ') and '.configsys-variant' in cmd:
                return Result(cmd, 0, stdout=f'via={self.marked}\ngpu=x\n')
            if 'describe' in cmd:
                return Result(cmd, 0, stdout='4.11.0\n')
            return Result(cmd, 0)

    for marked, winner in (('opencv-cuda12', 'opencv-cuda12'), ('opencv-build', 'opencv-build')):
        for cls in opencv.DRIVERS:
            v = cls(Fake(marked)).get_version(_RC(dir='opencv-git'))
            assert (v == '4.11.0') == (cls.name == winner), (cls.name, marked, v)


def test_probe_variant_from_cvconfig():
    from configsys.runner import Result

    class Cfg:
        '''No marker; cvconfig.h defines HAVE_CUDA and a cuda module links libcudnn.so.9 -> cuda12.'''
        def run(self, cmd, **kw):
            if cmd.startswith('ls ') and 'libopencv_core' in cmd:
                return Result(cmd, 0, stdout='libopencv_core.so\n')
            if cmd.startswith('cat '):
                return Result(cmd, 1)                        # no marker
            if 'grep -q "define HAVE_HIP"' in cmd:
                return Result(cmd, 1)
            if 'grep -q "define HAVE_CUDA"' in cmd or 'grep -q "define HAVE_CUDNN"' in cmd:
                return Result(cmd, 0)
            if cmd.startswith('ldd '):
                return Result(cmd, 0, stdout='libcudnn.so.9 => /lib/libcudnn.so.9\n')
            return Result(cmd, 0)

    d = next(c for c in opencv.DRIVERS if c.name == 'opencv-build')(Cfg())
    assert d.detected_variant(_RC(dir='opencv-git')) == 'opencv-cuda12'


@pytest.mark.skipif(_base_routes() is None, reason='base routes.hu not alongside the configsys package')
def test_pinned_stack_variants(monkeypatch):
    from configsys.routes import Resolver
    oh = str(pathlib.Path(__file__).resolve().parent.parent / 'opencv.hu')

    def stack(via, gpu, cuda):
        monkeypatch.setenv('CONFIGSYS_FACET_gpu', gpu)
        monkeypatch.setenv('CONFIGSYS_FACET_cuda', cuda)
        r = Resolver(_base_routes(), 'ubuntu', '24.04', 'x86_64',
                     pins={'opencv': via}, plugin_files=[(oh, 'plugin')])
        return set(r.resolve_names(['opencv']))

    n11 = stack('opencv-cuda11', 'nvidia', '11.5')   # CUDA 11 stack: cuda-toolkit-11 + cudnn-8 + gcc-10
    assert {'opencv-cuda11\\opencv', 'apt\\cuda-toolkit-11', 'script\\cudnn-8', 'gcc\\gcc-10'} <= n11
    assert 'script\\cudnn-9' not in n11 and 'script\\cuda-toolkit-12' not in n11

    n12 = stack('opencv-cuda12', 'nvidia', '12.4')   # CUDA 12 stack: cuda-toolkit-12 + cudnn-9, no gcc-10
    assert {'opencv-cuda12\\opencv', 'script\\cudnn-9', 'script\\cuda-toolkit-12'} <= n12
    assert 'gcc\\gcc-10' not in n12 and 'script\\cudnn-8' not in n12
    assert 'apt\\cuda-toolkit-11' not in n12                       # -11 is opt-in; only the 11 stack names it

    assert 'apt\\rocm-hip' in stack('opencv-hip', 'amd', '')    # AMD -> HIP
    cpu = stack('opencv-build', '', '')            # no GPU -> plain CPU source (+ cmake/git), no GPU deps
    assert 'opencv-build\\opencv' in cpu
    assert not any(x in k for k in cpu for x in ('cudnn', 'cuda-toolkit', 'gcc-10', 'rocm', 'cuda-repo'))
