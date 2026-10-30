param([int]$Port = 8794)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExecutable = Join-Path $projectRoot '.venv\Scripts\python.exe'
$runtimePackage = Join-Path $projectRoot 'tools\pi-runtime\node_modules\@earendil-works\pi-coding-agent\package.json'
if (-not (Test-Path -LiteralPath $runtimePackage)) {
    throw 'Run npm.cmd ci --prefix tools/pi-runtime --ignore-scripts --no-audit --no-fund before starting Pi.'
}
$outputDirectory = Join-Path $projectRoot 'outputs\pi_workflow_v1'
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
try {
    $existingService = Invoke-RestMethod "http://127.0.0.1:$Port/workflow/api/projects" -TimeoutSec 2
    if ($null -ne $existingService.projects) {
        Write-Output "Pi workflow is already available at http://127.0.0.1:$Port/workflow/"
        exit 0
    }
} catch { }
$process = Start-Process -FilePath $pythonExecutable -ArgumentList '-X','utf8','-m','video2scene.pi_workflow.server','--port',$Port -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $outputDirectory 'server.stdout.log') -RedirectStandardError (Join-Path $outputDirectory 'server.stderr.log') -PassThru
Write-Output "Starting Pi workflow (PID $($process.Id)): http://127.0.0.1:$Port/workflow/"
