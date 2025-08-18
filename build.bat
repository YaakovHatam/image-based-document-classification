setlocal EnableDelayedExpansion

rem === Script Params ===
set APP_NAME=AutoSignatureDetector
set ENTRY=main.py
set BUNDLE_DIR=%APP_NAME%
set VENV_DIR=.venv

rem Activate venv
call %VENV_DIR%\Scripts\activate

rem Install requirements.txt
if exist requirements.txt ( pip install -r requirements.txt )

rem Clean old builds
if exist %BUNDLE_DIR% ( rmdir /s /q %BUNDLE_DIR% )
if exist build ( rmdir /s /q build )
if exist dist ( rmdir /s /q dist )
if exist AutoSignatureDetector.spec ( del AutoSignatureDetector.spec )

rem Build (using PyInstaller create exec file)
PyInstaller --onefile --name %APP_NAME% %ENTRY%

rem Prepare bundle folder
if exist %BUNDLE_DIR% rmdir /s /q %BUNDLE_DIR%
mkdir %BUNDLE_DIR%

rem Copy executable
if exist dist\%APP_NAME%.exe (
  copy /Y dist\%APP_NAME%.exe %BUNDLE_DIR%\
) else if exist dist\%APP_NAME% (
  copy /Y dist\%APP_NAME% %BUNDLE_DIR%\
) else (
  echo ERROR: built executable not found in dist\
  exit /b 2
)

rem Copy config.toml
if exist config.toml (
  copy /Y config.toml %BUNDLE_DIR%\config.toml >nul
) else (
  echo Warning: config.toml not found in project root
)

rem Copy templates_pdf folder
if exist templates_pdf (
  xcopy templates_pdf %BUNDLE_DIR%\templates_pdf /E /I /Y >nul
) else (
  echo Warning: templates_pdf folder not found in project root
)

rem Create empty to_process folder
mkdir %BUNDLE_DIR%\to_process 2>nul

echo.
echo #############################################
echo Executable created check folder: %BUNDLE_DIR%
echo #############################################
echo.

pause


