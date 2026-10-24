$ErrorActionPreference = 'Stop'
Write-Host '=== NewsGuard AI: Windows setup ===' -ForegroundColor Cyan
if (-not (Test-Path '.venv')) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Force data, models, reports | Out-Null
if (-not (Test-Path 'data\news.csv')) {
  Write-Host 'No labelled training dataset found at data\news.csv.' -ForegroundColor Yellow
  Write-Host 'RAG works without a trained classifier. Add a labelled CSV before training RoBERTa.'
} else {
  Write-Host 'Found data\news.csv. Train the classifier with your verified column names:' -ForegroundColor Green
  Write-Host '.\.venv\Scripts\python.exe train_roberta.py --csv data\news.csv --text-col text --label-col label'
}
Write-Host 'Starting Streamlit...' -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m streamlit run app.py
