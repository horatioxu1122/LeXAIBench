param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path

Push-Location $repoRoot
try {
    $previousNativeErrorPreference = $null
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $previousNativeErrorPreference = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    if (-not $env:UV_CACHE_DIR) {
        $env:UV_CACHE_DIR = Join-Path $repoRoot ".uv-cache"
    }
    foreach ($threadVar in @(
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS"
    )) {
        if (-not (Get-Item -Path "Env:$threadVar" -ErrorAction SilentlyContinue)) {
            Set-Item -Path "Env:$threadVar" -Value "1"
        }
    }

    $uvOutput = $null
    $uvExit = 0
    try {
        $uvOutput = & uv run legalbench-xai @CliArgs 2>&1
        $uvExit = $LASTEXITCODE
    }
    catch {
        $uvOutput = $_.Exception.Message
        $uvExit = 1
    }
    if ($uvExit -eq 0) {
        if ($uvOutput) {
            $uvOutput
        }
        exit 0
    }

    $uvMessage = ($uvOutput | Out-String)
    $fallbackPython = $null
    foreach ($candidate in @(
        ".venv\\Scripts\\python.exe"
    )) {
        $resolved = Join-Path $repoRoot $candidate
        if (Test-Path $resolved) {
            $fallbackPython = $resolved
            break
        }
    }

    if (
        $fallbackPython -and
        (
            $uvMessage -match "Access is denied" -or
            $uvMessage -match "Failed to initialize cache"
        )
    ) {
        Write-Warning "uv run failed in the current shell; falling back to workspace-local python at $fallbackPython"
        $env:PYTHONPATH = "benchmark/src"
        & $fallbackPython -m legalbench_xai @CliArgs
        exit $LASTEXITCODE
    }

    exit $uvExit
}
finally {
    if ($null -ne $previousNativeErrorPreference) {
        $PSNativeCommandUseErrorActionPreference = $previousNativeErrorPreference
    }
    Pop-Location
}
