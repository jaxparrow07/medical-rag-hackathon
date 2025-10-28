# PowerShell Launcher for Medical RAG Frontend

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Medical RAG Frontend Server" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "  Please install Python 3.8 or higher" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if vector database exists
if (-not (Test-Path "data\vectordb\chroma.sqlite3")) {
    Write-Host ""
    Write-Host "⚠ Warning: Vector database not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please run setup_database.py first:" -ForegroundColor Yellow
    Write-Host "  python setup_database.py" -ForegroundColor White
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check document count
Write-Host "✓ Vector database found" -ForegroundColor Green

Write-Host ""
Write-Host "Starting server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Once started, open your browser to:" -ForegroundColor Yellow
Write-Host "  http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Start the server
python server.py

Read-Host "`nPress Enter to exit"
