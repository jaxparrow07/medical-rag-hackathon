@echo off
REM Frontend Launcher for Medical RAG System

echo.
echo ================================================================================
echo   Medical RAG Frontend Server
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if vector database exists
if not exist "data\vectordb\chroma.sqlite3" (
    echo Warning: Vector database not found!
    echo.
    echo Please run setup_database.py first:
    echo   python setup_database.py
    echo.
    pause
    exit /b 1
)

echo Starting server...
echo.
echo Once started, open your browser to:
echo   http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

REM Start the server
python server.py

pause
