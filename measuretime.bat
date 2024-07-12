@echo off
rem Get start time
for /F "tokens=1-4 delims=:.," %%a in ("%TIME%") do (
    set start_h=%%a
    set start_m=%%b
    set start_s=%%c
    set start_ms=%%d
)

rem Run the curl command
curl -X POST https://api-bpmn.azurewebsites.net/upload/ -F "file=@C:\Users\kofb\Documents\BPMN to XML\data\ex00_writer0097.jpg"

rem Get end time
for /F "tokens=1-4 delims=:.," %%a in ("%TIME%") do (
    set end_h=%%a
    set end_m=%%b
    set end_s=%%c
    set end_ms=%%d
)

rem Convert start time to milliseconds
set /A start_total_ms=(%start_h%*3600000) + (%start_m%*60000) + (%start_s%*1000) + %start_ms%
rem Convert end time to milliseconds
set /A end_total_ms=(%end_h%*3600000) + (%end_m%*60000) + (%end_s%*1000) + %end_ms%

rem Calculate duration in milliseconds
set /A duration_ms=end_total_ms-start_total_ms

rem Convert duration to seconds
set /A duration_s=duration_ms / 1000
set /A duration_ms=duration_ms %% 1000

rem Output the result
echo The command took %duration_s% seconds and %duration_ms% milliseconds to execute.
pause
