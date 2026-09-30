@echo off
rem MangaAR GUI launcher (Windows). Uses the project venv if present, else PATH.
rem Opens http://127.0.0.1:7860 (local only). Extra arguments go to "manga-arabic gui".
setlocal
set "ROOT=%~dp0.."
chcp 65001 >nul
if exist "%ROOT%\.venv\Scripts\manga-arabic.exe" (
  "%ROOT%\.venv\Scripts\manga-arabic.exe" gui %*
) else (
  where manga-arabic >nul 2>nul
  if errorlevel 1 (
    echo manga-arabic is not installed. See README.md ^(Install^).
    pause
    exit /b 1
  )
  manga-arabic gui %*
)
endlocal
