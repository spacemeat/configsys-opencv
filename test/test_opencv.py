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


def test_cuda_host_compiler_flag_no_nvcc(monkeypatch):
    # no nvcc on PATH -> no host-compiler override (never blocks / never guesses)
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    assert _drv()._cuda_host_compiler_flag() == ''


def test_driver_registration_shape():
    assert opencv.OpencvBuild.name == 'opencv-build'
    assert opencv.DRIVERS == [opencv.OpencvBuild]
