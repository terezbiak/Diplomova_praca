$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$allDstPath = Join-Path $PSScriptRoot "allDST_omni.csv"
$eventsPath = Join-Path $projectRoot "0_datasety\zoznam_udalosti_final.csv"
$outputPath = Join-Path $projectRoot "0_datasety\event_omni.csv"

if (-not (Test-Path $allDstPath)) {
    throw "Missing input file: $allDstPath"
}

if (-not (Test-Path $eventsPath)) {
    throw "Missing input file: $eventsPath"
}

$eventSpecs = Import-Csv $eventsPath |
    ForEach-Object {
        [pscustomobject]@{
            event_no = [int]$_.ID
            start    = [datetimeoffset]::Parse($_.Udalost_Od)
            end      = [datetimeoffset]::Parse($_.Udalost_Do)
        }
    } |
    Sort-Object start

if ($eventSpecs.Count -eq 0) {
    throw "No event intervals were found in $eventsPath"
}

for ($i = 1; $i -lt $eventSpecs.Count; $i++) {
    if ($eventSpecs[$i].start -le $eventSpecs[$i - 1].end) {
        throw "Overlapping event intervals detected between event_no $($eventSpecs[$i - 1].event_no) and $($eventSpecs[$i].event_no)."
    }
}

$rows = Import-Csv $allDstPath
$selected = New-Object System.Collections.Generic.List[object]
$eventIndex = 0

foreach ($row in $rows) {
    if ($eventIndex -ge $eventSpecs.Count) {
        break
    }

    $timestamp = [datetimeoffset]::Parse($row.time1)

    while ($eventIndex -lt $eventSpecs.Count -and $timestamp -gt $eventSpecs[$eventIndex].end) {
        $eventIndex++
    }

    if ($eventIndex -ge $eventSpecs.Count) {
        break
    }

    $event = $eventSpecs[$eventIndex]
    if ($timestamp -lt $event.start) {
        continue
    }

    $selected.Add([pscustomobject]@{
        time1    = $row.time1
        bz_gsm   = [double]$row.bz_gsm
        v        = [double]$row.v
        dst      = [double]$row.DST
        event_no = $event.event_no
    })
}

$selected | Export-Csv -Path $outputPath -NoTypeInformation -Encoding UTF8

$uniqueEvents = ($selected | Select-Object -ExpandProperty event_no | Sort-Object -Unique)

Write-Output ("Saved {0} rows to {1}" -f $selected.Count, $outputPath)
Write-Output ("Events covered: {0}" -f $uniqueEvents.Count)
Write-Output ("First timestamp: {0}" -f $selected[0].time1)
Write-Output ("Last timestamp: {0}" -f $selected[$selected.Count - 1].time1)
