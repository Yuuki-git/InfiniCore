"""Initialize the candidate's generated attention library once at package import."""

import ctypes
import os
from pathlib import Path


_core_library = None


def initialize():
    global _core_library
    switches = [os.environ.get(name, "0") for name in ("SS5_NT_ATTN", "SS5_NT_ADALN")]
    if any(value not in ("0", "1") for value in switches):
        raise ValueError("SS5_NT_ATTN and SS5_NT_ADALN must be 0 or 1")
    from .lib import _infinicore

    _core_library = ctypes.CDLL(str(Path(_infinicore.__file__).resolve(strict=True)))
    _core_library.ss5_nt_switches.restype = ctypes.c_uint
    _core_library.ss5_nt_calls.argtypes = (ctypes.c_uint,)
    _core_library.ss5_nt_calls.restype = ctypes.c_uint64
    expected = int(switches[0]) | (int(switches[1]) << 1)
    if _core_library.ss5_nt_switches() != expected:
        raise RuntimeError("Candidate switches differ from the values fixed at library initialization")
    if switches[0] == "1":
        cache = Path(os.environ["XDG_CACHE_HOME"]).resolve(strict=True)
        workspace = Path(os.environ["IFAB_WORKSPACE_ROOT"]).resolve(strict=True)
        if not cache.is_relative_to(workspace.parent):
            raise ValueError("Candidate driver cache escaped the web workspace and its parent")
        os.environ.setdefault("SS5_ROUND1_CACHE_SCOPE", str(cache))
        os.environ.setdefault("TRITON_CACHE_DIR", str(cache / "triton"))
        os.environ.setdefault("TRITON_DUMP_DIR", str(cache / "triton-dump"))
        os.environ.setdefault("TRITON_OVERRIDE_DIR", str(cache / "triton-override"))
        os.environ.setdefault("TRITON_CACHE_MANAGER", "infinicore._ss5_triton_cache:WorkspaceCacheManager")
        import torch
        import triton

        if torch.cuda.device_count() != 8:
            raise RuntimeError("The candidate requires eight gfx936 devices")
        configure = _core_library.ss5_configure_lds_ss5_s4_attention
        configure.argtypes = (ctypes.c_int, ctypes.c_uint)
        configure.restype = ctypes.c_int
        for device in range(8):
            if torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0] != "gfx936":
                raise RuntimeError(f"Unexpected target on device {device}")
            limit = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
            if type(limit) is not int or not 16384 <= limit <= 0xFFFF_FFFF:
                raise RuntimeError(f"Invalid attention LDS limit on device {device}: {limit!r}")
            status = configure(device, limit)
            if status:
                raise RuntimeError(f"Configuring attention LDS on device {device} returned {status}")


def snapshot():
    return {"switches": _core_library.ss5_nt_switches(),
            "calls": dict(zip(("attention", "adaln", "gate"),
                              (_core_library.ss5_nt_calls(index) for index in range(3)), strict=True))}
