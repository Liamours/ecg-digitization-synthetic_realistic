$out = "C:\researches\research-ecg-digitization\logs\resource_monitor_20260930.csv"
if (-not (Test-Path $out)) { "timestamp,cpu_pct,available_mb,gpu_pct" | Out-File $out -Encoding utf8 }
while ($true) {
    $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples.CookedValue
    $mem = (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
    $gpu = try { (Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage').CounterSamples.CookedValue | Measure-Object -Sum | Select-Object -ExpandProperty Sum } catch { 0 }
    "$(Get-Date -Format o),$([math]::Round($cpu,1)),$([math]::Round($mem,0)),$([math]::Round($gpu,1))" | Out-File $out -Append -Encoding utf8
    Start-Sleep -Seconds 15
}
