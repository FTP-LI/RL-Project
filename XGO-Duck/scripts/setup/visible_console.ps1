# Keep all subsequent project commands in one visible foreground console.
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Host.UI.RawUI.WindowTitle = 'XGO-Duck | Development console'
Set-Location -LiteralPath $projectRoot
$queueDir = Join-Path $projectRoot 'workspace/console'
New-Item -ItemType Directory -Force -Path $queueDir | Out-Null
$mutex = [Threading.Mutex]::new($false, 'Local\XGO-Duck-Visible-Console')
if (-not $mutex.WaitOne(0)) { Write-Host 'The project console is already open.'; exit 0 }
$recordDir = Join-Path $projectRoot ('records/deployment/' + (Get-Date -Format 'yyyy-MM-dd'))
New-Item -ItemType Directory -Force -Path $recordDir | Out-Null
$session = Get-Date -Format 'HHmmss'
Start-Transcript -Path (Join-Path $recordDir "console-$session-transcript.txt")
try {
    @{ pid = $PID; started_at = (Get-Date -Format o); state = 'idle' } | ConvertTo-Json | Set-Content (Join-Path $queueDir 'session.json')
    Write-Host 'Visible project console. Commands run here sequentially; close this window to stop.'
    while ($true) {
        $pending = Join-Path $queueDir 'pending.json'
        if (-not (Test-Path -LiteralPath $pending)) { Start-Sleep -Seconds 2; continue }
        $request = Get-Content -LiteralPath $pending -Raw | ConvertFrom-Json
        if ($request.id -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid command record identifier.' }
        $requestPath = Join-Path $recordDir ("console-request-" + $request.id + '.json')
        Move-Item -LiteralPath $pending -Destination $requestPath
        @{ pid = $PID; state = 'running'; request_id = $request.id; updated_at = (Get-Date -Format o) } | ConvertTo-Json | Set-Content (Join-Path $queueDir 'session.json')
        $result = [ordered]@{ id = $request.id; started_at = (Get-Date -Format o); request = $requestPath; status = 'running' }
        try {
            $commandArgs = @($request.arguments | ForEach-Object { [string]$_ })
            Write-Host ("COMMAND: " + $request.executable + ' ' + ($commandArgs | ConvertTo-Json -Compress))
            switch ($request.executable) {
                'wsl.exe' { & wsl.exe @commandArgs }
                'pwsh.exe' { & (Get-Command pwsh.exe).Source @commandArgs }
                default { throw 'Only project PowerShell or WSL commands are supported.' }
            }
            $result.exit_code = $LASTEXITCODE
            $result.status = if ($LASTEXITCODE -eq 0) { 'passed' } else { 'failed' }
        } catch {
            $result.status = 'failed'
            $result.error = $_.Exception.Message
            Write-Host $result.error -ForegroundColor Red
        } finally {
            $result.finished_at = (Get-Date -Format o)
            $result | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $recordDir ("console-result-" + $request.id + '.json'))
            Write-Host ("RESULT: " + $result.status + '. Waiting for the next project command in this same window.')
            @{ pid = $PID; state = 'idle'; last_request_id = $request.id; updated_at = (Get-Date -Format o) } | ConvertTo-Json | Set-Content (Join-Path $queueDir 'session.json')
        }
    }
} finally {
    @{ pid = $PID; finished_at = (Get-Date -Format o); state = 'closed' } | ConvertTo-Json | Set-Content (Join-Path $queueDir 'session.json')
    Stop-Transcript
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
