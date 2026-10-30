param(
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$PipelineArgs
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
function Convert-ToWslPath([string]$Value) {
    if ($Value -match '^[A-Za-z]:[\\/]') {
        return '/mnt/' + $Value.Substring(0,1).ToLower() + '/' + $Value.Substring(3).Replace('\','/')
    }
    return $Value
}
$LinuxRoot = Convert-ToWslPath $ProjectRoot
$ConvertedArgs = @($PipelineArgs | ForEach-Object { Convert-ToWslPath $_ })
& wsl -d Ubuntu-22.04 --exec bash "$LinuxRoot/scripts/run_wsl.sh" @ConvertedArgs
exit $LASTEXITCODE
