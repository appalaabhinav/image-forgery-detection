# ==============================================================================
# Helper Script to Push Image Forgery Detection Web App to GitHub
# ==============================================================================

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  AI-Based Image Forgery Detection - GitHub Deploy Helper" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

$repoUrl = Read-Host "Paste your GitHub Repository URL (e.g., https://github.com/appalaabhinav/image-forgery-detection.git)"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "Error: No repository URL provided." -ForegroundColor Red
    exit 1
}

# Ensure git remote is configured
git remote remove origin 2>$null
git remote add origin $repoUrl

Write-Host ""
Write-Host "Pushing code and model weights to GitHub (Branch: main)..." -ForegroundColor Yellow
Write-Host "Note: Uploading ~100MB of model weights. This might take 1-2 minutes depending on your internet speed." -ForegroundColor DarkGray
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host "  SUCCESS! Code and weights pushed to GitHub." -ForegroundColor Green
    Write-Host "==========================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next Step: Deploy on Streamlit Cloud (Takes 1 Minute)" -ForegroundColor Cyan
    Write-Host "1. Open https://share.streamlit.io in your browser."
    Write-Host "2. Click 'Create app'."
    Write-Host "3. Select your repository: $repoUrl"
    Write-Host "4. Main file path: app.py"
    Write-Host "5. Click 'Deploy!' - Your app will be live on any device!"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "Push failed. Check your GitHub permissions or credentials." -ForegroundColor Red
}
