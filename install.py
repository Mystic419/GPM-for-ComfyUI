from __future__ import annotations

import importlib.util
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements.txt"
LLAMA_IMPORT_NAME = "llama_cpp"
LLAMA_PIP_NAME = "llama-cpp-python"
LLAMA_CUBLAS_INDEX_BASE = "https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels"
INSTALL_MODE_ENV = "GPM_LLAMA_INSTALL_MODE"
INSTALL_MODE_AUTO = "auto"
INSTALL_MODE_CPU = "cpu"
INSTALL_MODE_CUDA = "cuda"
VALID_INSTALL_MODES = {INSTALL_MODE_AUTO, INSTALL_MODE_CPU, INSTALL_MODE_CUDA}
INTERNAL_SUPPORT_MODULE = "gpm_vlm_internal_multimodal"


def _log(message: str) -> None:
    print(f"[GPM install] {message}")


def _run(args: list[str]) -> bool:
    _log(f"Running: {' '.join(args)}")
    completed = subprocess.run(args, check=False)
    if completed.returncode == 0:
        return True
    _log(f"Command failed (exit={completed.returncode})")
    return False


def _pip_install(args: list[str]) -> bool:
    cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check", "install", *args]
    return _run(cmd)


def _read_install_mode() -> str:
    mode = str(os.environ.get(INSTALL_MODE_ENV, "") or "").strip().casefold()
    if mode in VALID_INSTALL_MODES:
        return mode
    if mode:
        _log(f"Unknown {INSTALL_MODE_ENV} value '{mode}', using '{INSTALL_MODE_AUTO}'.")
    return INSTALL_MODE_AUTO


def _llama_installed() -> bool:
    return importlib.util.find_spec(LLAMA_IMPORT_NAME) is not None


def _llama_version() -> str:
    try:
        import llama_cpp  # type: ignore

        return str(getattr(llama_cpp, "__version__", "") or "").strip() or "<unknown>"
    except Exception:
        return "<unknown>"


def _probe_internal_support() -> tuple[bool, str]:
    try:
        importlib.import_module(INTERNAL_SUPPORT_MODULE)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _install_requirements() -> bool:
    if not REQUIREMENTS_PATH.exists():
        _log(f"No requirements file found at {REQUIREMENTS_PATH}; skipping.")
        return True

    lines = [
        line.strip()
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        _log("requirements.txt is empty; skipping normal requirements install.")
        return True

    _log("Installing normal requirements from requirements.txt")
    return _pip_install(["-r", str(REQUIREMENTS_PATH)])


def _detect_cuda() -> tuple[bool, str]:
    if importlib.util.find_spec("torch") is None:
        return False, ""

    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return False, ""

        cuda_version = str(torch.version.cuda or "").strip()
        if not cuda_version:
            return False, ""

        return True, "cu" + cuda_version.replace(".", "")
    except Exception:
        return False, ""


def _detect_avx_folder() -> str:
    if importlib.util.find_spec("cpuinfo") is None:
        return "AVX"

    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info() or {}
        flags = info.get("flags") or []
        if "avx2" in flags:
            return "AVX2"
    except Exception:
        pass
    return "AVX"


def _install_llama_cuda(cuda_tag: str, avx_folder: str) -> bool:
    index_url = f"{LLAMA_CUBLAS_INDEX_BASE}/{avx_folder}/{cuda_tag}"
    _log(f"Trying CUDA/cuBLAS wheel index: {index_url}")
    return _pip_install([
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        LLAMA_PIP_NAME,
        f"--index-url={index_url}",
    ])


def _install_llama_fallback() -> bool:
    _log("Falling back to plain pip install for llama-cpp-python.")
    return _pip_install(["--upgrade", LLAMA_PIP_NAME])


def main() -> int:
    _log("Starting install flow.")
    _log(f"Python: {sys.executable}")
    _log(f"Platform: {platform.platform()}")

    if not _install_requirements():
        _log("requirements.txt install failed.")
        return 1

    mode = _read_install_mode()
    cuda_available, cuda_tag = _detect_cuda()
    avx_folder = _detect_avx_folder()
    _log(f"Install mode: {mode}")
    _log(f"CUDA detected by torch: {'YES' if cuda_available else 'NO'}")
    if cuda_tag:
        _log(f"Detected CUDA tag: {cuda_tag}")

    llama_ready = _llama_installed()
    if llama_ready:
        _log(f"llama_cpp already installed (version={_llama_version()}).")
    else:
        use_cuda = False
        if mode == INSTALL_MODE_CUDA:
            use_cuda = True
        elif mode == INSTALL_MODE_CPU:
            use_cuda = False
        else:
            use_cuda = cuda_available

        installed = False
        if use_cuda and cuda_tag:
            installed = _install_llama_cuda(cuda_tag=cuda_tag, avx_folder=avx_folder)
            if not installed:
                _log("CUDA wheel install failed; falling back to standard pip install.")
        elif use_cuda and not cuda_tag:
            _log("CUDA mode requested but CUDA version tag is unavailable; using standard pip install.")

        if not installed:
            installed = _install_llama_fallback()

        llama_ready = installed and _llama_installed()
        if llama_ready:
            _log(f"llama_cpp installed successfully (version={_llama_version()}).")
        else:
            _log("llama_cpp is still not importable after install attempts.")
            return 1

    internal_ok, internal_error = _probe_internal_support()
    _log(f"Internal support import ({INTERNAL_SUPPORT_MODULE}): {'OK' if internal_ok else 'FAILED'}")
    if not internal_ok:
        _log(f"Internal support detail: {internal_error}")
        return 1

    _log("Install flow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
