while ($true) {
    Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'src.digitize.run' } | ForEach-Object {
        $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
        if ($p -and $p.PriorityClass -ne 'BelowNormal') { $p.PriorityClass = 'BelowNormal' }
    }
    Start-Sleep -Seconds 20
}
