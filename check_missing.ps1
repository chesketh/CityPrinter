$presets = Get-Content 'c:\Users\chesk\.cursor-tutor\CityBuilder\frontend\src\data\presets.json' | ConvertFrom-Json
$glbs = (Get-ChildItem 'c:\Users\chesk\.cursor-tutor\CityBuilder\output\*.glb').BaseName
$missing = $presets | Where-Object { $glbs -notcontains $_.id }
$found = $presets | Where-Object { $glbs -contains $_.id }
Write-Host "Total presets: $($presets.Count)"
Write-Host "Presets with GLB: $($found.Count)"
Write-Host "Missing: $(($missing | Measure-Object).Count)"
Write-Host ""
Write-Host "Missing presets:"
foreach ($m in $missing) {
    Write-Host "  $($m.id)"
}
