from __future__ import annotations

import importlib.util
import importlib
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements.txt"
LLAMA_IMPORT_NAME = "llama_cpp"
LLAMA_PIP_NAME = "llama-cpp-python"
# TODO: Keep this aligned to the currently tested GPM llama-cpp-python runtime version until hosted wheel manifests land.
LLAMA_PIP_VERSION = "0.3.22"
LLAMA_PIP_SPEC = f"{LLAMA_PIP_NAME}=={LLAMA_PIP_VERSION}"
LLAMA_CUBLAS_INDEX_BASE = "https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels"
# TODO: Support a GPM-hosted wheel manifest so supported CUDA families can be updated without editing installer code.
SUPPORTED_CUDA_WHEEL_TAGS = {"cu121", "cu122", "cu123", "cu124", "cu125"}
PHASE1_LOCAL_WHEEL_CUDA_TAG = "cu130"
GPM_WHEEL_RELEASE_BASE_URL = "https://github.com/Mystic419/GPM-for-ComfyUI/releases/download/llama-cpp-python-gpm-cu130-v0.3.22"
PHASE1_LOCAL_WHEEL_SUPPORTED_ARCHES = {
    "sm86": {"family": "RTX 30-series / Ampere", "status": "validated"},
    "sm89": {"family": "RTX 40-series / Ada", "status": "built_unvalidated"},
    "sm120": {"family": "RTX 50-series / Blackwell", "status": "built_unvalidated"},
}
PHASE1_HOSTED_WHEEL_SHA256 = {
    "sm86": "806DBB36DB10415D3B8515C7D4D4F9F61959600572F30AA71E901380EE144349",
    "sm89": "76164B230902243581DEC2252815C597DF5EB9077AD9CA72E740833E54615B63",
    "sm120": "890DA6CF9B302D9108F7924A81D818FFABBD36914D995E816AE4F8670CE227A6",
}
PHASE1_LOCAL_WHEEL_PLATFORM_TAG = "win_amd64"
PHASE1_LOCAL_WHEEL_PYTHON_TAG = "cp312"
PHASE1_LOCAL_WHEEL_ALLOW_UNKNOWN_ARCH_FALLBACK = False
SOURCE_BUILD_TIMEOUT_SECONDS = 90 * 60
INSTALL_MODE_ENV = "GPM_LLAMA_INSTALL_MODE"
INSTALL_MODE_AUTO = "auto"
INSTALL_MODE_CPU = "cpu"
INSTALL_MODE_CUDA = "cuda"
INSTALL_MODE_CUDA_BUILD = "cuda-build"
VALID_INSTALL_MODES = {INSTALL_MODE_AUTO, INSTALL_MODE_CPU, INSTALL_MODE_CUDA, INSTALL_MODE_CUDA_BUILD}
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
    timeout_seconds: int | None = None,
) -> tuple[bool, str, bool]:
    _log(f"Running: {' '.join(args)}")
    if not stream_output:
        completed = subprocess.run(args, check=False, env=env)
        if completed.returncode == 0:
            return True, "", False
        _log(f"Command failed (exit={completed.returncode})")
        return False, f"pip exited with code {completed.returncode}", False

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
        return False, str(exc), False

    assert process.stdout is not None
    read_done = threading.Event()

    def _reader() -> None:
        nonlocal recent_lines
        try:
            for line in process.stdout:
                print(line, end="")
                stripped = line.strip()
                if stripped:
                    recent_lines.append(stripped)
                    if len(recent_lines) > 300:
                        recent_lines = recent_lines[-300:]
        finally:
            read_done.set()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    timed_out = False
    while True:
        if process.poll() is not None:
            break
        if timeout_seconds is not None and (time.time() - started_at) >= timeout_seconds:
            timed_out = True
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
            break
        time.sleep(0.2)

    try:
        process.wait(timeout=10)
    except Exception:
        pass
    read_done.wait(timeout=1.0)
    reader_thread.join(timeout=1.0)
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=1.0)

    if timed_out:
        return False, f"timed out after {_format_elapsed(timeout_seconds or 0)}", True

    if process.returncode == 0:
        return True, "", False

    _log(f"Command failed (exit={process.returncode})")
    return False, _extract_failure_detail(recent_lines, process.returncode), False


def _pip_install(args: list[str], env: dict[str, str] | None = None) -> bool:
    cmd = [sys.executable, "-m", "pip", "--disable-pip-version-check", "install", *args]
    ok, _, _ = _run(cmd, env=env)
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


def _detect_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def _detect_platform_tag() -> str:
    if os.name == "nt" and platform.machine().lower() in {"amd64", "x86_64"}:
        return "win_amd64"
    return f"{sys.platform}_{platform.machine().lower()}"


def _detect_cuda_sm_arch() -> str:
    if importlib.util.find_spec("torch") is None:
        return ""
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return ""
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm{major}{minor}"
    except Exception:
        return ""


def _phase1_raw_pip_wheel_filename() -> str:
    return f"llama_cpp_python-{LLAMA_PIP_VERSION}-py3-none-{PHASE1_LOCAL_WHEEL_PLATFORM_TAG}.whl"


def _phase1_archived_wheel_filename(python_tag: str, cuda_tag: str, sm_arch: str) -> str:
    return (
        f"llama_cpp_python-{LLAMA_PIP_VERSION}-{python_tag}-{PHASE1_LOCAL_WHEEL_PLATFORM_TAG}-"
        f"{cuda_tag}-{sm_arch}.whl"
    )


def _is_phase1_pip_valid_wheel_filename(name: str) -> bool:
    return name == _phase1_raw_pip_wheel_filename()


def _find_phase1_local_wheel_candidate(python_tag: str, cuda_tag: str, sm_arch: str) -> tuple[Path | None, str]:
    wheel_dirs = [
        SCRIPT_DIR / "wheels",
        SCRIPT_DIR / "dist" / "gpm_wheels",
    ]
    archived_name = _phase1_archived_wheel_filename(python_tag=python_tag, cuda_tag=cuda_tag, sm_arch=sm_arch)
    raw_name = _phase1_raw_pip_wheel_filename()

    for wheel_dir in wheel_dirs:
        if not wheel_dir.exists():
            continue
        archived_path = wheel_dir / archived_name
        if archived_path.exists():
            return archived_path, "archived"
    for wheel_dir in wheel_dirs:
        if not wheel_dir.exists():
            continue
        raw_path = wheel_dir / raw_name
        if raw_path.exists():
            return raw_path, "raw"
    return None, ""


def _install_local_wheel_via_pip_filename(wheel_path: Path) -> bool:
    if _is_phase1_pip_valid_wheel_filename(wheel_path.name):
        _log("Installing GPM CUDA wheel with --no-deps to avoid changing shared ComfyUI dependencies.")
        return _pip_install(["--force-reinstall", "--no-deps", str(wheel_path)])

    _log(f"Preparing archived wheel for pip install: {wheel_path}")
    pip_wheel_name = _phase1_raw_pip_wheel_filename()
    with tempfile.TemporaryDirectory(prefix="gpm_llama_wheel_") as temp_dir:
        temp_wheel_path = Path(temp_dir) / pip_wheel_name
        shutil.copy2(wheel_path, temp_wheel_path)
        _log(f"Temporary pip wheel path: {temp_wheel_path}")
        _log("Installing GPM CUDA wheel with --no-deps to avoid changing shared ComfyUI dependencies.")
        return _pip_install(["--force-reinstall", "--no-deps", str(temp_wheel_path)])


def _download_file(url: str, dest_path: Path) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url) as response, dest_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        return True, ""
    except urllib.error.URLError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def _sha256_upper(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().upper()


def _install_llama_local_phase1_wheel(cuda_tag: str) -> bool:
    _log("Looking for local GPM CUDA wheel (Phase 1).")
    if os.name != "nt":
        _log("Local GPM wheel path is currently Windows-only; skipping.")
        return False
    python_tag = _detect_python_tag()
    if python_tag != PHASE1_LOCAL_WHEEL_PYTHON_TAG:
        _log(f"Local GPM wheel path supports Python tag {PHASE1_LOCAL_WHEEL_PYTHON_TAG} only (detected {python_tag}); skipping.")
        return False
    platform_tag = _detect_platform_tag()
    if platform_tag != PHASE1_LOCAL_WHEEL_PLATFORM_TAG:
        _log(f"Local GPM wheel path supports platform tag {PHASE1_LOCAL_WHEEL_PLATFORM_TAG} only (detected {platform_tag}); skipping.")
        return False
    if cuda_tag != PHASE1_LOCAL_WHEEL_CUDA_TAG:
        _log(f"Local GPM wheel path supports CUDA tag {PHASE1_LOCAL_WHEEL_CUDA_TAG} only (detected {cuda_tag}); skipping.")
        return False

    sm_arch = _detect_cuda_sm_arch()
    if sm_arch:
        arch_info = PHASE1_LOCAL_WHEEL_SUPPORTED_ARCHES.get(sm_arch)
        if not arch_info:
            _log(f"No local GPM wheel is supported for detected GPU arch {sm_arch}; skipping local wheel.")
            return False
        family = str(arch_info.get("family", "unknown family"))
        status = str(arch_info.get("status", "unknown"))
        _log(f"Detected GPU arch {sm_arch} ({family}), local wheel status={status}.")
        if status != "validated":
            _log(
                "This local wheel is built but not maintainer-validated; community validation is needed. "
                "Continuing because detected arch matches."
            )
    elif PHASE1_LOCAL_WHEEL_ALLOW_UNKNOWN_ARCH_FALLBACK:
        _log(
            "GPU arch detection unavailable; allowing Phase 1 local wheel fallback for known tested profile "
            f"({PHASE1_LOCAL_WHEEL_CUDA_TAG}/sm86)."
        )
    else:
        _log(
            "GPU arch detection unavailable; skipping local Phase 1 wheel install because this wheel is validated "
            "only for sm86 (RTX 30-series / Ampere)."
        )
        return False

    # TODO: Replace filename-only matching with manifest metadata once hosted wheel manifests are implemented.
    wheel_path, wheel_kind = _find_phase1_local_wheel_candidate(
        python_tag=python_tag,
        cuda_tag=cuda_tag,
        sm_arch=sm_arch,
    )
    if not wheel_path:
        _log("No matching local GPM CUDA wheel found in ./wheels or ./dist/gpm_wheels.")
        return False

    if wheel_kind == "archived":
        _log(f"Matched local GPM archived wheel: {wheel_path}")
    else:
        _log(f"Matched local GPM raw pip wheel fallback: {wheel_path}")

    ok = _install_local_wheel_via_pip_filename(wheel_path)
    if ok:
        _log("Local GPM CUDA wheel install succeeded.")
    else:
        _log("Local GPM CUDA wheel install failed.")
    return ok


def _install_llama_hosted_phase1_wheel(cuda_tag: str) -> bool:
    _log("Looking for hosted GPM CUDA wheel.")
    if os.name != "nt":
        _log("Hosted GPM wheel path is currently Windows-only; skipping.")
        return False
    python_tag = _detect_python_tag()
    if python_tag != PHASE1_LOCAL_WHEEL_PYTHON_TAG:
        _log(f"Hosted GPM wheel path supports Python tag {PHASE1_LOCAL_WHEEL_PYTHON_TAG} only (detected {python_tag}); skipping.")
        return False
    platform_tag = _detect_platform_tag()
    if platform_tag != PHASE1_LOCAL_WHEEL_PLATFORM_TAG:
        _log(f"Hosted GPM wheel path supports platform tag {PHASE1_LOCAL_WHEEL_PLATFORM_TAG} only (detected {platform_tag}); skipping.")
        return False
    if cuda_tag != PHASE1_LOCAL_WHEEL_CUDA_TAG:
        _log(f"Hosted GPM wheel path supports CUDA tag {PHASE1_LOCAL_WHEEL_CUDA_TAG} only (detected {cuda_tag}); skipping.")
        return False

    sm_arch = _detect_cuda_sm_arch()
    if not sm_arch:
        _log("GPU arch detection unavailable; skipping hosted GPM wheel install.")
        return False
    arch_info = PHASE1_LOCAL_WHEEL_SUPPORTED_ARCHES.get(sm_arch)
    if not arch_info:
        _log(f"No hosted GPM wheel is supported for detected GPU arch {sm_arch}; skipping hosted wheel.")
        return False
    family = str(arch_info.get("family", "unknown family"))
    status = str(arch_info.get("status", "unknown"))
    _log(f"Detected GPU arch {sm_arch} ({family}), hosted wheel status={status}.")
    if status != "validated":
        _log("This hosted wheel is built but not maintainer-validated; community validation is needed. Continuing because detected arch matches.")

    archived_name = _phase1_archived_wheel_filename(python_tag=python_tag, cuda_tag=cuda_tag, sm_arch=sm_arch)
    wheel_url = f"{GPM_WHEEL_RELEASE_BASE_URL}/{archived_name}"
    _log(f"Hosted wheel URL: {wheel_url}")
    with tempfile.TemporaryDirectory(prefix="gpm_llama_hosted_wheel_") as temp_dir:
        temp_archived_path = Path(temp_dir) / archived_name
        _log("Hosted wheel download started.")
        download_ok, download_error = _download_file(url=wheel_url, dest_path=temp_archived_path)
        if not download_ok:
            _log(f"Hosted wheel download failed: {download_error}")
            return False
        _log("Hosted wheel download completed.")

        expected_sha = PHASE1_HOSTED_WHEEL_SHA256.get(sm_arch, "")
        if expected_sha:
            actual_sha = _sha256_upper(temp_archived_path)
            if actual_sha == expected_sha.upper():
                _log(f"SHA256 verified: {actual_sha}")
            else:
                _log(f"SHA256 failed for hosted wheel: expected {expected_sha.upper()} got {actual_sha}")
                return False

        ok = _install_local_wheel_via_pip_filename(temp_archived_path)
        if ok:
            _log("Hosted GPM CUDA wheel install succeeded.")
        else:
            _log("Hosted GPM CUDA wheel install failed.")
        return ok


def _install_llama_cuda(cuda_tag: str, avx_folder: str) -> bool:
    index_url = f"{LLAMA_CUBLAS_INDEX_BASE}/{avx_folder}/{cuda_tag}"
    _log(f"Trying CUDA/cuBLAS wheel index: {index_url}")
    return _pip_install([
        "--upgrade",
        "--force-reinstall",
        "--no-deps",
        LLAMA_PIP_SPEC,
        f"--index-url={index_url}",
    ])


def _install_llama_cuda_source_build() -> tuple[bool, str, bool]:
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
        LLAMA_PIP_SPEC,
    ]
    started_at = time.time()
    ok, detail, timed_out = _run(
        cmd,
        env=env,
        stream_output=True,
        heartbeat_message="still building llama-cpp-python from source...",
        heartbeat_interval_seconds=25,
        timeout_seconds=SOURCE_BUILD_TIMEOUT_SECONDS,
    )
    elapsed = _format_elapsed(time.time() - started_at)
    if ok:
        _log(f"local CUDA build finished in {elapsed}")
        return True, "", False
    if timed_out:
        _log(f"CUDA source build timed out after {_format_elapsed(SOURCE_BUILD_TIMEOUT_SECONDS)}")
        _log("Continuing with CPU fallback install; CPU mode may be slower.")
        return False, detail, True
    _log(f"local CUDA build failed after {elapsed}")
    return False, detail, False


def _install_llama_fallback() -> bool:
    _log("Falling back to plain pip install for llama-cpp-python.")
    return _pip_install(["--upgrade", LLAMA_PIP_SPEC])


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
        if mode in {INSTALL_MODE_CUDA, INSTALL_MODE_CUDA_BUILD}:
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
                    _log("CUDA wheel install failed.")
            else:
                if mode == INSTALL_MODE_AUTO:
                    _log(f"Detected CUDA tag '{cuda_tag}' is outside supported prebuilt wheel families.")
                    _log("Local CUDA source build is available but not enabled in auto mode because it can take a long time.")
                    _log("To enable it, set GPM_LLAMA_INSTALL_MODE=cuda-build and reinstall.")
                else:
                    _log(f"Detected CUDA tag '{cuda_tag}' is outside supported prebuilt wheel families ({', '.join(sorted(SUPPORTED_CUDA_WHEEL_TAGS))}).")
                    _log("Local CUDA source build is available only in cuda-build mode.")

            if not installed:
                installed = _install_llama_local_phase1_wheel(cuda_tag=cuda_tag)
                if not installed:
                    _log("Continuing without local GPM CUDA wheel.")
            if not installed:
                installed = _install_llama_hosted_phase1_wheel(cuda_tag=cuda_tag)
                if not installed:
                    _log("Continuing without hosted GPM CUDA wheel.")

            should_try_source_build = (not installed) and (mode == INSTALL_MODE_CUDA_BUILD)
            if should_try_source_build:
                _log("cuda-build mode is enabled; attempting local CUDA source build before CPU fallback.")
                source_ok, source_error, source_timed_out = _install_llama_cuda_source_build()
                installed = source_ok
                if not source_ok and not source_timed_out:
                    _log(f"CUDA source build failed: {source_error}")
                    _log("Continuing with CPU fallback install; CPU mode may be slower.")
        elif mode == INSTALL_MODE_CUDA_BUILD and use_cuda and not cuda_tag:
            _log("cuda-build mode requested but CUDA version tag is unavailable; attempting local CUDA source build anyway.")
            source_ok, source_error, source_timed_out = _install_llama_cuda_source_build()
            installed = source_ok
            if not source_ok and not source_timed_out:
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
