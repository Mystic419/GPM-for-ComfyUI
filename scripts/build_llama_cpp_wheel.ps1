param(
    [string]$PythonExe = "",
    [string]$Version = "0.3.22",
    [string]$CudaArchitectures = "86",
    [string]$OutputDir = ".\\dist\\gpm_wheels"
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    Write-Host "[GPM wheel build] $Message"
}

function Normalize-ArchTag {
    param([string]$Value)
    $valueOrEmpty = $Value
    if ($null -eq $valueOrEmpty) {
        $valueOrEmpty = ""
    }
    $trimmed = $valueOrEmpty.Trim()
    if ($trimmed -match '^\d+$') {
        return "sm$trimmed"
    }
    $safe = ($trimmed -replace '[^A-Za-z0-9]+', '_').Trim('_').ToLowerInvariant()
    if (-not $safe) {
        return "sm_unknown"
    }
    if ($safe -notmatch '^sm') {
        return "sm$safe"
    }
    return $safe
}

$startTime = Get-Date

try {
    if ([string]::IsNullOrWhiteSpace($PythonExe)) {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
    }

    $resolvedPython = (Resolve-Path -LiteralPath $PythonExe).Path
    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputDir)

    if (-not (Test-Path -LiteralPath $resolvedOutput)) {
        New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
    }

    $env:CMAKE_ARGS = "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
    $env:FORCE_CMAKE = "1"

    Write-Log "Starting local llama-cpp-python wheel build."
    Write-Log "Python executable: $resolvedPython"
    Write-Log "Target version: $Version"
    Write-Log "CUDA architectures: $CudaArchitectures"
    Write-Log "Output directory: $resolvedOutput"
    Write-Log "CMAKE_ARGS=$($env:CMAKE_ARGS)"
    Write-Log "FORCE_CMAKE=$($env:FORCE_CMAKE)"
    Write-Log "Running pip wheel (this may take a while)..."

    $wheelSpec = "llama-cpp-python==$Version"
    & $resolvedPython -m pip wheel --no-cache-dir --no-binary=llama-cpp-python --wheel-dir $resolvedOutput $wheelSpec
    if ($LASTEXITCODE -ne 0) {
        throw "Wheel build failed with exit code $LASTEXITCODE."
    }

    $rawWheel = Get-ChildItem -LiteralPath $resolvedOutput -Filter "llama_cpp_python-$Version-*.whl" |
        Where-Object { $_.Name -like "*win_amd64.whl" } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $rawWheel) {
        throw "Wheel build succeeded but no raw llama_cpp_python wheel was found in $resolvedOutput."
    }

    $pyInfo = & $resolvedPython -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to detect Python tag from $resolvedPython."
    }
    $pythonTagLine = $pyInfo | Select-Object -First 1
    if ($null -eq $pythonTagLine) {
        $pythonTagLine = ""
    }
    $pythonTag = $pythonTagLine.Trim()
    if (-not $pythonTag) {
        throw "Detected empty Python tag from $resolvedPython."
    }

    $platformTag = "win_amd64"
    # TODO: Detect CUDA tag from torch/runtime metadata when this becomes stable in installer tooling.
    $cudaTag = "cu130"
    $archTag = Normalize-ArchTag -Value $CudaArchitectures
    $archivedName = "llama_cpp_python-$Version-$pythonTag-$platformTag-$cudaTag-$archTag.whl"
    $archivedPath = Join-Path -Path $resolvedOutput -ChildPath $archivedName
    $archivedExists = Test-Path -LiteralPath $archivedPath
    if ($archivedExists) {
        Write-Log "Archived artifact already exists and will be overwritten: $archivedPath"
    }
    Copy-Item -LiteralPath $rawWheel.FullName -Destination $archivedPath -Force
    $archivedInfo = Get-Item -LiteralPath $archivedPath
    $archivedSizeBytes = [int64]$archivedInfo.Length
    $archivedSizeMB = [Math]::Round($archivedSizeBytes / 1MB, 2)

    $elapsed = (Get-Date) - $startTime
    Write-Log "Wheel build completed successfully."
    Write-Log "Raw wheel path: $($rawWheel.FullName)"
    Write-Log "Archived wheel path: $archivedPath"
    Write-Log "Archived wheel size: $archivedSizeBytes bytes ($archivedSizeMB MB)"
    Write-Log ("Elapsed time: {0:hh\:mm\:ss}" -f $elapsed)
    Write-Log "Use the archived wheel artifact for distribution/testing, not the raw pip wheel filename."
    exit 0
}
catch {
    $elapsed = (Get-Date) - $startTime
    Write-Error "[GPM wheel build] $($_.Exception.Message)"
    Write-Host ("[GPM wheel build] Elapsed time: {0:hh\:mm\:ss}" -f $elapsed)
    exit 1
}
