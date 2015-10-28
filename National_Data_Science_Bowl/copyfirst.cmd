echo off
xcopy /Y %1 %2 /T /E

dir %1 /b /s /A:D >tempfolderlist.txt

for /f "tokens=1 delims=¬" %%a in (./tempfolderlist.txt) do (

    dir "%%a" /b /A:-D >tempfilelist.txt

    setlocal enabledelayedexpansion

    set counter=0

    for /f "tokens=1 delims=¬" %%b in (./tempfilelist.txt) do (

        IF !counter! LSS 10 call :docopy %1 "%%a\%%b" %2
        set /a counter+=1

    )

    endlocal
)

del /q tempfolderlist.txt
del /q tempfilelist.txt
GOTO:EOF

:docopy
set sourcePath=%~1
set sourceFile=%~2
set targetPath=%~3
set sourceNoDrive=%sourceFile:~3,5000%
set sourcePathNoDrive=%sourcePath:~3,5000%
set sourceNoDrive=!sourceNoDrive:%sourcePathNoDrive%\=!

copy "%sourceFile%" "%targetPath%\%sourceNoDrive%" >> out.txt

GOTO:EOF