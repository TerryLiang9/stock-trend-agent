# Stock L2 Collector - Windows Scheduled Task Setup
# Creates a task to run collector.py every weekday at 9:25 AM

$taskName = "StockL2Collector"
$pythonExe = "d:\code\codex\stock_agent\stock_agent\.venv\Scripts\python.exe"
$scriptDir = "d:\code\codex\stock_agent\stock_agent\scripts"

$action = New-ScheduledTaskAction `
    -Execute $pythonExe `
    -Argument "collector.py --env DEV" `
    -WorkingDirectory $scriptDir

$trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At 09:25

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force

Write-Host "Task '$taskName' created successfully."
Write-Host "  Schedule: Mon-Fri at 09:25"
Write-Host "  Command:  $pythonExe collector.py --env DEV"
Write-Host "  Working:  $scriptDir"
Write-Host "  On battery: allowed"
Write-Host "  Multiple instances: ignored"
