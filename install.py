from __future__ import annotations

import importlib.util
import importlib
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements.txt"
LLAMA_IMPORT_NAME = "llama_cpp"
LLAMA_PIP_NAME = "llama-cpp-python"
LLAMA_CUBLAS_INDEX_BASE = "https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels"
SUPPORTED_CUDA_WHEEL_TAGS = {"cu121", "cu122", "cu123", "cu124", "cu125"}
INSTALL_MODE_ENV = "GPM_LLAMA_INSTALL_MODE"
INSTALL_MODE_AUTO = "auto"
INSTALL_MODE_CPU = "cpu"
INSTALL_MODE_CUDA = "cuda"
VALID_INSTALL_MODES = {INSTALL_MODE_AUTO, INSTALL_MODE_CPU, INSTALL_MODE_CUDA}
INTERNAL_SUPPORT_MODULE = "gpm_vlm_internal_multimodal"


def _log(message: str) -> None:
    print(f"[GPM install] {message}")


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _extract_failure_detail(lines: list[str], returncode: int) -> str:
    detail_lines = [line.strip() for line in lines if line and line.strip()]
    error_lines = [line for line in detail_lines if "error" in line.casefold()]
    if error_lines:
        return error_lines[-1]
    if detail_lines:
        return detail_lines[-1]
    return f"pip exited with code {returncode}"


def _run(
    args: list[str],
    env: dict[str, str] | None = None,
    stream_output: bool = False,
    heartbeat_message: str | None = None,
    heartbeat_interval_seconds: int = 25,
) -> tuple[bool, str]:
    _log(f"Running: {' '.join(args)}")
    if not stream_output:
        completed = subprocess.run(args, check=False, env=env)
        if completed.returncode == 0:
            return True, ""
        _log(f"Command failed (exit={completed.returncode})")
        return False, f"pip exited with code {completed.returncode}"

    started_at = time.time()
    recent_lines: list[str] = []
    stop_heartbeat = threading.Event()

    def _heartbeat() -> None:
        while not stop_heartbeat.wait(max(1, heartbeat_interval_seconds)):
            elapsed = _format_elapsed(time.time() - started_at)
            if heartbeat_message:
                _log(heartbeat_message)
            _log(f"elapsed: {elapsed}")

    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()

    try:
        process = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)
        _log(f"Command failed to start: {exc}")
        return False, str(exc)

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        stripped = line.strip()
        if stripped:
            recent_lines.append(stripped)
            if len(recent_lines) > 300:
                recent_lines = recent_lines[-300:]

    process.wait()
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=1.0)

    if process.returncode == 0:
        return True, ""

    _log(f"Command failed (exit={process.returncode})")
    return False, _extract_failure_detail(recent_lines, process.returncode)


def _pip_install(args: list[str], env: dict[str, str] | None = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check", "install", *args]
    ok, _ = _run(cmd, env=env)
    return ok


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


def _install_llama_cuda_source_build() -> tuple[bool, str]:
    _log("No usable prebuilt CUDA wheel was found for this system.")
    _log("local CUDA build started")
    _log("this may take several minutes")
    _log("the window may appear quiet between compiler steps")
    _log("Attempting local CUDA source build for llama-cpp-python (this may take a few minutes).")
    _log("If CUDA build fails, installer will continue with CPU fallback.")

    env = os.environ.copy()
    existing_cmake_args = str(env.get("CMAKE_ARGS", "") or "").strip()
    cuda_flag = "-DGGML_CUDA=on"
    if cuda_flag not in existing_cmake_args:
        env["CMAKE_ARGS"] = f"{existing_cmake_args} {cuda_flag}".strip()
    env["FORCE_CMAKE"] = "1"

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--no-binary=llama-cpp-python",
        LLAMA_PIP_NAME,
    ]
    started_at = time.time()
    ok, detail = _run(
        cmd,
        env=env,
        stream_output=True,
        heartbeat_message="still building llama-cpp-python from source...",
        heartbeat_interval_seconds=25,
    )
    elapsed = _format_elapsed(time.time() - started_at)
    if ok:
        _log(f"local CUDA build finished in {elapsed}")
        return True, ""
    _log(f"local CUDA build failed after {elapsed}")
    return False, detail


def _install_llama_fallback() -> bool:
    _log("Falling back to plain pip install for llama-cpp-python.")
    return _pip_install(["--upgrade", LLAMA_PIP_NAME])


def _probe_llama_capabilities() -> tuple[str, str]:
    try:
        import llama_cpp  # type: ignore
    except Exception as exc:
        return "<unknown>", f"not importable ({exc})"

    version = str(getattr(llama_cpp, "__version__", "") or "").strip() or "<unknown>"

    try:
        probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        if callable(probe):
            return version, "CUDA capable (llama_supports_gpu_offload=yes)" if probe() else "CPU only (llama_supports_gpu_offload=no)"

        inner = getattr(llama_cpp, "llama_cpp", None)
        inner_probe = getattr(inner, "llama_supports_gpu_offload", None)
        if callable(inner_probe):
            return version, "CUDA capable (llama_cpp.llama_supports_gpu_offload=yes)" if inner_probe() else "CPU only (llama_cpp.llama_supports_gpu_offload=no)"

        return version, "unknown (GPU capability probe unavailable)"
    except Exception as exc:
        return version, f"unknown (GPU capability probe failed: {exc})"


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
            if cuda_tag in SUPPORTED_CUDA_WHEEL_TAGS:
                _log(f"Detected CUDA tag '{cuda_tag}' is in supported prebuilt wheel families.")
                installed = _install_llama_cuda(cuda_tag=cuda_tag, avx_folder=avx_folder)
                if not installed:
                    _log("CUDA wheel install failed; attempting CUDA source build before CPU fallback.")
            else:
                _log(f"Detected CUDA tag '{cuda_tag}' is outside supported prebuilt wheel families ({', '.join(sorted(SUPPORTED_CUDA_WHEEL_TAGS))}).")

            if not installed:
                source_ok, source_error = _install_llama_cuda_source_build()
                installed = source_ok
                if not source_ok:
                    _log(f"CUDA source build failed: {source_error}")
                    _log("Continuing with CPU fallback install; CPU mode may be slower.")
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

    detected_version, capability = _probe_llama_capabilities()
    _log(f"llama_cpp post-install probe: version={detected_version}, capability={capability}")

    internal_ok, internal_error = _probe_internal_support()
    _log(f"Internal support import ({INTERNAL_SUPPORT_MODULE}): {'OK' if internal_ok else 'FAILED'}")
    if not internal_ok:
        _log(f"Internal support detail: {internal_error}")
        return 1

    _log("Install flow complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
