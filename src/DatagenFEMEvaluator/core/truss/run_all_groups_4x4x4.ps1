param(
  [int]$Workers = 18,
  [int]$Samples = 25000,
  [int]$BasicSize = 4,
  [int]$PollSeconds = 10,
  [int]$IdleTimeoutMinutes = 15,
  [int]$GroupTimeoutMinutes = 180,
  [bool]$StopOnFailure = $true,
  [string[]]$IncludeGroups = @(),
  [string[]]$ExcludeGroups = @()
)

$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$auto = Join-Path $root 'auto_generate_4x4x4.py'
$dbPath = (Get-ChildItem -Path $root -File -Filter '*.json' |
  Sort-Object Length -Descending |
  Select-Object -First 1 -ExpandProperty FullName)
if ([string]::IsNullOrWhiteSpace($dbPath)) {
  throw "No .json file found under $root"
}
$outRoot = Join-Path $projectRoot 'workspace'
$batchDir = Join-Path $outRoot '_batch'

function Write-DebugLine {
  param([string]$Message)
  $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  Write-Host "[$ts] $Message"
}

function Test-BasicSizeCompatible {
  param(
    [double[]]$LatticeLengths,
    [int]$TargetBasicSize
  )
  if ($null -eq $LatticeLengths -or $LatticeLengths.Count -ne 3) {
    return @{ Ok = $true; Detail = "lattice_lengths_missing_or_invalid" }
  }

  $axes = @("x", "y", "z")
  for ($k = 0; $k -lt 3; $k++) {
    $L = [double]$LatticeLengths[$k]
    if ($L -le 0) {
      return @{ Ok = $false; Detail = "axis=$($axes[$k]) invalid_length=$L" }
    }
    $ratio = [double]$TargetBasicSize / $L
    $ratioRound = [math]::Round($ratio)
    if ([math]::Abs($ratio - $ratioRound) -gt 1e-9) {
      return @{
        Ok = $false
        Detail = "basic_size=$TargetBasicSize incompatible_with_lattice_lengths=[$($LatticeLengths -join ',')] axis=$($axes[$k]) ratio=$ratio"
      }
    }
  }
  return @{ Ok = $true; Detail = "ok" }
}

# -------- Runtime knobs --------
$workers = $Workers
$samples = $Samples
$pollSeconds = $PollSeconds
$idleTimeoutMinutes = $IdleTimeoutMinutes
$groupTimeoutMinutes = $GroupTimeoutMinutes
$excludeGroups = $ExcludeGroups
$includeGroups = $IncludeGroups
$stopFlag = Join-Path $batchDir 'STOP'

New-Item -ItemType Directory -Force -Path $batchDir | Out-Null
if (Test-Path $stopFlag) { Remove-Item -Force $stopFlag }

$db = Get-Content -Path $dbPath -Raw | ConvertFrom-Json
$groups = @($db.groups.PSObject.Properties.Name | Sort-Object)
if ($includeGroups.Count -gt 0) {
  $allow = @($includeGroups | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' })
  $groups = @($groups | Where-Object { $allow -contains $_ })
}
if ($excludeGroups.Count -gt 0) {
  $groups = @($groups | Where-Object { $excludeGroups -notcontains $_ })
}
$groupsFile = Join-Path $batchDir 'groups_to_run.txt'
$groups | Set-Content -Path $groupsFile -Encoding UTF8
$total = $groups.Count

$summary = Join-Path $batchDir 'progress.tsv'
"timestamp	index	total	group	exit_code	status" | Out-File -FilePath $summary -Encoding UTF8

Write-DebugLine "Batch launcher file: $PSCommandPath"
Write-DebugLine "Running python script: $auto"
Write-DebugLine "Group DB json: $dbPath"
Write-DebugLine "Output root: $outRoot"
Write-DebugLine "Progress file: $summary"
Write-DebugLine "Config: workers=$workers, samples=$samples, basicSize=$BasicSize, poll=${pollSeconds}s, idleTimeout=${idleTimeoutMinutes}m, groupTimeout=${groupTimeoutMinutes}m"

$globalStop = $false
$doneGroups = 0
$skippedGroups = New-Object System.Collections.Generic.List[object]
for ($i = 0; $i -lt $total; $i++) {
  if ($globalStop) { break }

  $g = $groups[$i].Trim()
  if ([string]::IsNullOrWhiteSpace($g)) { continue }

  $runDir = Join-Path $outRoot $g
  if (Test-Path $runDir) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Add-Content -Path $summary -Value ("$ts`t$($i+1)`t$total`t$g`t0`tSKIP_DIR_EXISTS")
    $skippedGroups.Add([PSCustomObject]@{ group = $g; reason = "DIR_EXISTS"; detail = $runDir }) | Out-Null
    $doneGroups += 1
    $pendingGroups = $total - $doneGroups
    $skipStatus = "current=$g | done=$doneGroups/$total | pending=$pendingGroups | SKIP_DIR_EXISTS"
    Write-Progress -Id 1 -Activity "Batch groups" -Status $skipStatus -PercentComplete ([int](100 * $doneGroups / $total))
    Write-DebugLine "SKIP group=$g reason=DIR_EXISTS path=$runDir"
    continue
  }

  $lattice = $null
  try {
    $lattice = @($db.groups.$g.lattice_lengths)
  } catch {
    $lattice = $null
  }
  $compat = Test-BasicSizeCompatible -LatticeLengths $lattice -TargetBasicSize $BasicSize
  if (-not $compat.Ok) {
    $ts = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Add-Content -Path $summary -Value ("$ts`t$($i+1)`t$total`t$g`t0`tSKIP_INCOMPATIBLE_BASIC_SIZE")
    $skippedGroups.Add([PSCustomObject]@{ group = $g; reason = "INCOMPATIBLE_BASIC_SIZE"; detail = $compat.Detail }) | Out-Null
    $doneGroups += 1
    $pendingGroups = $total - $doneGroups
    $skipStatus = "current=$g | done=$doneGroups/$total | pending=$pendingGroups | SKIP_INCOMPATIBLE_BASIC_SIZE"
    Write-Progress -Id 1 -Activity "Batch groups" -Status $skipStatus -PercentComplete ([int](100 * $doneGroups / $total))
    Write-DebugLine "SKIP group=$g reason=INCOMPATIBLE_BASIC_SIZE detail=$($compat.Detail)"
    continue
  }
  New-Item -ItemType Directory -Force -Path $runDir | Out-Null

  $idx = "{0:D3}" -f ($i + 1)
  $log = Join-Path $batchDir ("$idx`_$g.log")
  $err = Join-Path $batchDir ("$idx`_$g.err.log")
  $csv = Join-Path $runDir ("$g-architecture.csv")
  $abaqusDir = Join-Path $runDir 'abaqus_txt'
  $crystalDir = Join-Path $runDir 'crystal_4x4x4'

  $startAt = Get-Date
  $lastProgressAt = $startAt
  $lastSize = if (Test-Path $csv) { (Get-Item $csv).Length } else { -1 }
  $lastLogSize = if (Test-Path $log) { (Get-Item $log).Length } else { -1 }
  $lastErrSize = if (Test-Path $err) { (Get-Item $err).Length } else { -1 }
  $lastAbaqusCount = if (Test-Path $abaqusDir) { (Get-ChildItem $abaqusDir -File -Filter '*.txt' -ErrorAction SilentlyContinue | Measure-Object).Count } else { -1 }
  $lastCrystalCount = if (Test-Path $crystalDir) { (Get-ChildItem $crystalDir -File -Filter '*.txt' -ErrorAction SilentlyContinue | Measure-Object).Count } else { -1 }

  $start = $startAt.ToString('yyyy-MM-dd HH:mm:ss')
  Add-Content -Path $summary -Value ("$start`t$($i+1)`t$total`t$g`t`tSTART")
  Write-DebugLine "START group=$g index=$($i+1)/$total"

  $args = @(
    '-u', $auto, $g,
    '--basic-size', $BasicSize,
    '--samples', $samples,
    '--workers', $workers,
    '--resume',
    '--run-dir', $runDir,
    '--group-db', $dbPath
  )
  Write-DebugLine "CMD: python $($args -join ' ')"

  $proc = Start-Process -FilePath python -ArgumentList $args -WorkingDirectory $root `
    -RedirectStandardOutput $log -RedirectStandardError $err -PassThru
  Write-DebugLine "PROCESS started pid=$($proc.Id) log=$log err=$err"

  $finalStatus = ''
  $finalCode = ''
  $loopCount = 0

  while ($true) {
    Start-Sleep -Seconds $pollSeconds

    $hadProgress = $false
    $loopCount += 1

    if (Test-Path $stopFlag) {
      if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
      $finalStatus = 'STOPPED'
      $finalCode = 130
      $globalStop = $true
      Write-DebugLine "STOP flag detected. Stopping group=$g pid=$($proc.Id)"
      break
    }

    if (Test-Path $csv) {
      $sz = (Get-Item $csv).Length
      if ($sz -gt $lastSize) {
        $lastSize = $sz
        $hadProgress = $true
      }
    }

    if (Test-Path $log) {
      $logSize = (Get-Item $log).Length
      if ($logSize -gt $lastLogSize) {
        $lastLogSize = $logSize
        $hadProgress = $true
      }
    }

    if (Test-Path $err) {
      $errSize = (Get-Item $err).Length
      if ($errSize -gt $lastErrSize) {
        $lastErrSize = $errSize
        $hadProgress = $true
      }
    }

    if (Test-Path $abaqusDir) {
      $abaqusCount = (Get-ChildItem $abaqusDir -File -Filter '*.txt' -ErrorAction SilentlyContinue | Measure-Object).Count
      if ($abaqusCount -gt $lastAbaqusCount) {
        $lastAbaqusCount = $abaqusCount
        $hadProgress = $true
      }
    }

    if (Test-Path $crystalDir) {
      $crystalCount = (Get-ChildItem $crystalDir -File -Filter '*.txt' -ErrorAction SilentlyContinue | Measure-Object).Count
      if ($crystalCount -gt $lastCrystalCount) {
        $lastCrystalCount = $crystalCount
        $hadProgress = $true
      }
    }

    if ($hadProgress) {
      $lastProgressAt = Get-Date
    }

    $now = Get-Date
    $elapsedMin = [math]::Round(($now - $startAt).TotalMinutes, 1)
    $idleMin = [math]::Round(($now - $lastProgressAt).TotalMinutes, 1)
    $pendingGroups = $total - $doneGroups
    $pctGroups = [int](100 * $doneGroups / $total)

    $csvSizeMB = if (Test-Path $csv) { [math]::Round(((Get-Item $csv).Length / 1MB), 2) } else { 0 }
    $abaqusCountNow = if (Test-Path $abaqusDir) { (Get-ChildItem $abaqusDir -File -Filter '*.txt' -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }
    $crystalCountNow = if (Test-Path $crystalDir) { (Get-ChildItem $crystalDir -File -Filter '*.txt' -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }

    $stage = "constraints/csv"
    if ($abaqusCountNow -gt 0) { $stage = "abaqus_txt" }
    if ($crystalCountNow -gt 0) { $stage = "crystal_4x4x4" }

    $status = "current=$g | done=$doneGroups/$total | pending=$pendingGroups | stage=$stage | elapsed=${elapsedMin}m idle=${idleMin}m csv=${csvSizeMB}MB abaqus=$abaqusCountNow crystal=$crystalCountNow pid=$($proc.Id)"
    Write-Progress -Id 1 -Activity "Batch groups" -Status $status -PercentComplete $pctGroups

    if ($hadProgress -or ($loopCount % 6 -eq 0)) {
      Write-DebugLine "HEARTBEAT $status"
    }

    if (($now - $startAt).TotalMinutes -gt $groupTimeoutMinutes) {
      if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
      $finalStatus = 'TIMEOUT'
      $finalCode = 124
      Write-DebugLine "TIMEOUT group=$g elapsed=${elapsedMin}m limit=${groupTimeoutMinutes}m"
      break
    }

    if (($now - $lastProgressAt).TotalMinutes -gt $idleTimeoutMinutes) {
      if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
      $finalStatus = 'IDLE_TIMEOUT'
      $finalCode = 125
      Write-DebugLine "IDLE_TIMEOUT group=$g idle=${idleMin}m limit=${idleTimeoutMinutes}m"
      break
    }

    if ($proc.HasExited) {
      $finalCode = $proc.ExitCode
      $finalStatus = if ($proc.ExitCode -eq 0) { 'DONE' } else { 'FAIL' }
      Write-DebugLine "EXIT group=$g pid=$($proc.Id) code=$finalCode status=$finalStatus"
      break
    }
  }

  $end = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
  Add-Content -Path $summary -Value ("$end`t$($i+1)`t$total`t$g`t$finalCode`t$finalStatus")
  $doneGroups += 1
  $pendingGroups = $total - $doneGroups
  Write-Progress -Id 1 -Activity "Batch groups" -Status ("current=$g | done=$doneGroups/$total | pending=$pendingGroups | status=$finalStatus") -PercentComplete ([int](100 * $doneGroups / $total))
  Write-DebugLine "END group=$g status=$finalStatus code=$finalCode"

  if ($StopOnFailure -and $finalStatus -ne 'DONE') {
    Write-DebugLine "STOP_ON_FAILURE triggered by group=$g"
    $globalStop = $true
  }
}

Write-Progress -Id 1 -Activity "Batch groups" -Completed
if ($skippedGroups.Count -gt 0) {
  $skipReport = Join-Path $batchDir "skipped_groups.tsv"
  "group`treason`tdetail" | Out-File -FilePath $skipReport -Encoding UTF8
  foreach ($row in $skippedGroups) {
    Add-Content -Path $skipReport -Value ("$($row.group)`t$($row.reason)`t$($row.detail)")
  }
  Write-DebugLine "Skipped groups total=$($skippedGroups.Count)"
  Write-DebugLine "Skip list file: $skipReport"
  foreach ($row in $skippedGroups) {
    Write-DebugLine ("SKIPPED group={0} reason={1} detail={2}" -f $row.group, $row.reason, $row.detail)
  }
} else {
  Write-DebugLine "Skipped groups total=0"
}
Write-DebugLine "Batch finished."
