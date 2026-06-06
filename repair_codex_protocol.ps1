$ErrorActionPreference = "Stop"

$manifest = "C:\Program Files\WindowsApps\OpenAI.Codex_26.527.3686.0_x64__2p2nqsd0c76g0\AppxManifest.xml"
$log = "C:\Users\Dell\finetuned\repair_codex_protocol.log"

"$(Get-Date -Format o) Waiting for Codex processes to close..." | Out-File -LiteralPath $log -Encoding utf8

while (Get-Process -Name "Codex","codex" -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 2
}

"$(Get-Date -Format o) Re-registering Codex AppX package..." | Out-File -LiteralPath $log -Append -Encoding utf8
Add-AppxPackage -DisableDevelopmentMode -Register $manifest

"$(Get-Date -Format o) Checking codex:// protocol registration..." | Out-File -LiteralPath $log -Append -Encoding utf8
$check = & reg query HKCR\codex /s 2>&1
$check | Out-File -LiteralPath $log -Append -Encoding utf8

"$(Get-Date -Format o) Done." | Out-File -LiteralPath $log -Append -Encoding utf8
