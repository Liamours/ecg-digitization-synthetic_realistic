# Canonical pipeline on every MAC 400 page of one dataset: final labels and one picture per page.
# Runs in the GPU environment (scripts/setup_gpu_env.sh): text reading, grid search and trace network on the GPU,
# about 6 s per page. Starts a detached process at below-normal priority, so it survives the shell that launched it.
# Only pages whose manifest page_style is mac400 are run. Resumable: run it again and finished pages are skipped.
# Progress: results/inferences/canonical_<dataset>/progress.csv and logs/digitize-canonical_<dataset>-<date>.log (ETA per page).
# Usage (PowerShell, from anywhere): repo\ecg-digitization-synthetic_realistic\scripts\run_canonical.ps1 -Dataset esta
param(
    [string]$Dataset = "mac400-phone_photo",   # alias from configs/paths.yml
    [string]$Layout = "mac400_phone"           # mac400_phone takes dot pitches of 6 to 16 px, below 300 dpi
)
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "..\..\logs"
$python = Join-Path $repo "..\..\.venvs\digitize_gpu\Scripts\python.exe"
$name = "canonical_$Dataset"
$env:PYTHONPATH = $repo
$run = @("-m", "src.digitize.run", "--config", "configs/digitize.yml", "--layout", $Layout,
         "--out-dir", "@inferences/$name", "--final-only", "--device", "cuda",
         "--manifest", "@$Dataset/_labels/manifest.csv", "--style", "mac400", "@$Dataset")
$p = Start-Process -FilePath $python -ArgumentList $run -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logs "$name.out.txt") -RedirectStandardError (Join-Path $logs "$name.err.txt")
$p.PriorityClass = "BelowNormal"
"started process $($p.Id) at below-normal priority"
