# Canonical pipeline on every MAC 400 page of ecg-mac400-phone_photo: final labels and one picture per page.
# Runs in the GPU environment (scripts/setup_gpu_env.sh): text reading, grid search and trace network on the GPU,
# about 6 s per page. Starts a detached process at below-normal priority, so it survives the shell that launched it.
# Resumable: run it again and finished pages are skipped.
# Progress: results/inferences/canonical_mac400-phone_photo/progress.csv and logs/digitize-mac400_phone-<date>.log (ETA per page).
# Usage (PowerShell, from anywhere): repo\ecg-digitization-synthetic_realistic\scripts\run_phone_photos.ps1
$repo = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $repo "..\..\logs"
$python = Join-Path $repo "..\..\.venvs\digitize_gpu\Scripts\python.exe"
$env:PYTHONPATH = $repo
$run = @("-m", "src.digitize.run", "--config", "configs/digitize.yml", "--layout", "mac400_phone",
         "--out-dir", "@inferences/canonical_mac400-phone_photo", "--final-only", "--device", "cuda",
         "--manifest", "@mac400-phone_photo/_labels/manifest.csv", "--style", "mac400", "@mac400-phone_photo")
$p = Start-Process -FilePath $python -ArgumentList $run -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logs "run_phone_photos.out.txt") -RedirectStandardError (Join-Path $logs "run_phone_photos.err.txt")
$p.PriorityClass = "BelowNormal"
"started process $($p.Id) at below-normal priority"
