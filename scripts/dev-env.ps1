# Dot-source this in PowerShell before working:  . .\scripts\dev-env.ps1
# Points every cache, the interpreter and the virtualenv at one data directory, so that several
# gigabytes of corpus, model weights and build caches do not land on the system drive.
# Set ORDERORDER_DATA_DIR to choose where that is; it defaults to .\data inside the repo.
if ($env:ORDERORDER_DATA_DIR) { $data = $env:ORDERORDER_DATA_DIR }
else { $data = Join-Path (Get-Location).Path "data" }
foreach ($d in @("", "uv-cache", "uv-python", "hf", "corpus", "postgres", "ollama")) {
    $p = if ($d) { Join-Path $data $d } else { $data }
    if (-not (Test-Path $p)) { New-Item -ItemType Directory -Force $p | Out-Null }
}
$env:ORDERORDER_DATA_DIR    = $data
$env:UV_CACHE_DIR           = Join-Path $data "uv-cache"
$env:UV_PYTHON_INSTALL_DIR  = Join-Path $data "uv-python"
$env:UV_PROJECT_ENVIRONMENT = Join-Path $data "venv"
$env:HF_HOME                = Join-Path $data "hf"
$env:OLLAMA_MODELS          = Join-Path $data "ollama"
$env:PYTHONUTF8             = "1"
$env:PYTHONIOENCODING       = "utf-8"
# The virtualenv holds one editable install, and `uv sync` writes into it the absolute path of
# whichever checkout ran it last. A second checkout of this repository -- a git worktree, a clone
# beside it -- then gets an `orderorder` on the PATH that runs the *first* checkout's code, and it
# fails in the way that costs the most time: the command exists, and the subcommand just added to it
# does not. `pyproject.toml` pins this for pytest; this is the same fix for the CLI.
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
Write-Host "OrderOrder dev environment: data and caches under $data"
Write-Host "                            source read from $env:PYTHONPATH"
Write-Host "Next: uv sync        (installs into $data\venv)"
Write-Host "      uv run orderorder --help"
