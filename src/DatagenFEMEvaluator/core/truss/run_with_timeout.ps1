param(
  [Parameter(Mandatory = $true)]
  [string]$Command,
  [int]$TimeoutSeconds = 30,
  [string]$WorkingDirectory = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
  $WorkingDirectory = (Get-Location).Path
}

$out = Join-Path $WorkingDirectory "_last_timeout_test.out.log"
$err = Join-Path $WorkingDirectory "_last_timeout_test.err.log"

if (Test-Path $out) { Remove-Item -Force $out }
if (Test-Path $err) { Remove-Item -Force $err }

$proc = Start-Process -FilePath "pwsh" `
  -ArgumentList @("-NoProfile", "-Command", $Command) `
  -WorkingDirectory $WorkingDirectory `
  -RedirectStandardOutput $out `
  -RedirectStandardError $err `
  -PassThru

$finished = $proc.WaitForExit($TimeoutSeconds * 1000)
if (-not $finished) {
  Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
  Write-Output "TIMEOUT: killed after ${TimeoutSeconds}s"
  exit 124
}

Write-Output "EXIT_CODE: $($proc.ExitCode)"
if (Test-Path $out) {
  Get-Content $out -Tail 80
}
if (Test-Path $err) {
  $errText = Get-Content $err -Raw
  if (-not [string]::IsNullOrWhiteSpace($errText)) {
    Write-Output "----- STDERR -----"
    Write-Output $errText
  }
}
exit $proc.ExitCode
