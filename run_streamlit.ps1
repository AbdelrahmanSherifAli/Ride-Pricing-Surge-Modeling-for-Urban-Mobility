# PowerShell script to run Streamlit app
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Ride Price & Surge Prediction App" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if streamlit is installed
try {
    python -c "import streamlit" 2>$null
    Write-Host "Streamlit is installed. Starting app..." -ForegroundColor Green
} catch {
    Write-Host "Streamlit is not installed. Installing..." -ForegroundColor Yellow
    pip install streamlit
    Write-Host ""
}

# Run streamlit
Write-Host "Starting Streamlit server..." -ForegroundColor Green
Write-Host "The app will open in your browser automatically." -ForegroundColor Yellow
Write-Host ""

streamlit run streamlit_app.py
