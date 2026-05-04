param(
    [string]$PythonExe = "",
    [Parameter(Mandatory = $true)]
    [string]$WheelPath,
    [switch]$ForceReinstall
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    Write-Host "[GPM wheel test] $Message"
}

$startTime = Get-Date

try {
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }

    $resolvedPython = (Resolve-Path -LiteralPath $PythonExe).Path
    $resolvedWheel = (Resolve-Path -LiteralPath $WheelPath).Path

    $wheelInfo = Get-Item -LiteralPath $resolvedWheel -ErrorAction Stop
    $wheelSizeBytes = [int64]$wheelInfo.Length
    $wheelSizeMB = [Math]::Round($wheelSizeBytes / 1MB, 2)

    Write-Log "Starting local wheel install test."
    Write-Log "Python executable: $resolvedPython"
    Write-Log "Wheel path: $resolvedWheel"
    Write-Log "Wheel file size: $wheelSizeBytes bytes ($wheelSizeMB MB)"

    $pipArgs = @("-m", "pip", "install")
    if ($ForceReinstall) {
        $pipArgs += "--force-reinstall"
        Write-Log "ForceReinstall: enabled"
    } else {
        Write-Log "ForceReinstall: disabled"
    }
    $pipArgs += $resolvedWheel

    Write-Log "Installing wheel with pip (no source build flags)..."
    & $resolvedPython @pipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed with exit code $LASTEXITCODE."
    }

    $probeScript = @'
import os
import sys

def _bool_to_text(value):
    if value is True:
        return "True"
    if value is False:
        return "False"
    return "unknown"

try:
    import llama_cpp
except Exception as exc:
    print(f"IMPORT_ERROR={exc}")
    raise

version = getattr(llama_cpp, "__version__", "unknown")
print(f"LLAMA_CPP_VERSION={version}")

top_probe = getattr(llama_cpp, "llama_supports_gpu_offload", None)
nested_probe = None
inner = getattr(llama_cpp, "llama_cpp", None)
if inner is not None:
    nested_probe = getattr(inner, "llama_supports_gpu_offload", None)

top_result = None
nested_result = None

if callable(top_probe):
    try:
        top_result = bool(top_probe())
    except Exception:
        top_result = None
print(f"GPU_OFFLOAD_TOP={_bool_to_text(top_result)}")

if callable(nested_probe):
    try:
        nested_result = bool(nested_probe())
    except Exception:
        nested_result = None
print(f"GPU_OFFLOAD_NESTED={_bool_to_text(nested_result)}")

sysinfo_text = ""
print_fn = getattr(llama_cpp, "llama_print_system_info", None)
if callable(print_fn):
    try:
        raw = print_fn()
        if isinstance(raw, bytes):
            sysinfo_text = raw.decode("utf-8", errors="ignore")
        else:
            sysinfo_text = str(raw)
    except Exception as exc:
        sysinfo_text = f"error:{exc}"
else:
    inner_print = getattr(inner, "llama_print_system_info", None) if inner is not None else None
    if callable(inner_print):
        try:
            raw = inner_print()
            if isinstance(raw, bytes):
                sysinfo_text = raw.decode("utf-8", errors="ignore")
            else:
                sysinfo_text = str(raw)
        except Exception as exc:
            sysinfo_text = f"error:{exc}"

cuda_detected = False
if sysinfo_text:
    lowered = sysinfo_text.lower()
    cuda_detected = ("cuda" in lowered) or ("cublas" in lowered) or ("ggml_cuda" in lowered)
print(f"SYSTEM_INFO_CUDA_HINT={_bool_to_text(cuda_detected)}")
if sysinfo_text:
    snippet = " ".join(sysinfo_text.split())
    if len(snippet) > 220:
        snippet = snippet[:220] + "..."
    print(f"SYSTEM_INFO_SNIPPET={snippet}")
'@

    # Execute probe from a temp file to avoid PowerShell multiline quoting issues with python -c.
    $probeTempFile = Join-Path -Path ([System.IO.Path]::GetTempPath()) -ChildPath ("gpm_llama_probe_{0}.py" -f ([Guid]::NewGuid().ToString("N")))
    try {
        [System.IO.File]::WriteAllText($probeTempFile, $probeScript, [System.Text.UTF8Encoding]::new($false))
        Write-Log "Running post-install probe (import/version/GPU/system-info)..."
        & $resolvedPython $probeTempFile
        if ($LASTEXITCODE -ne 0) {
            throw "Post-install probe failed (llama_cpp import/probe error)."
        }
    }
    finally {
        if (Test-Path -LiteralPath $probeTempFile) {
            Remove-Item -LiteralPath $probeTempFile -Force -ErrorAction SilentlyContinue
        }
    }

    $elapsed = (Get-Date) - $startTime
    Write-Log "Wheel install test completed successfully."
    Write-Log ("Elapsed time: {0:hh\:mm\:ss}" -f $elapsed)
    exit 0
}
catch {
    $elapsed = (Get-Date) - $startTime
    Write-Error "[GPM wheel test] $($_.Exception.Message)"
    Write-Host ("[GPM wheel test] Elapsed time: {0:hh\:mm\:ss}" -f $elapsed)
    exit 1
}
