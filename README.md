# NewsGuard AI — Fake News Detection + Agentic RAG

A local Windows Streamlit project based on the supplied abstract, **Fake News Detection Using NLP with RoBERTa Algorithm**, extended with lightweight Python agents and a retrieval-augmented news Q&A assistant.

## Features

- RoBERTa fine-tuning workflow for labelled REAL/FAKE news CSVs.
- Stratified train/validation/test split; no training on the held-out test set.
- Evaluation: Accuracy, precision, recall, F1, specificity, FPR, FNR, MCC, Cohen's Kappa, ROC-AUC, confusion matrix, classification report.
- News classifier agent that loads a local fine-tuned model and reports its actual output; does not make up a prediction if no model is available.
- Live News & Source Finder using GDELT public news search (no API key) or optional NewsAPI (API key required).
- Publisher/domain, headline, date metadata, and original article links; selected public article URLs can be fetched and indexed for RAG.
- RAG ingestion from pasted text, public article URLs, TXT/MD/PDF/CSV files.
- TF-IDF retrieval from the user's indexed sources.
- Optional Ollama explanation agent that must answer from retrieved evidence. If Ollama is unavailable, the app shows retrieved passages rather than fabricating an answer.
- SQLite prediction/source history, Plotly analytics, CSV and JSON downloads.

## Important limits

- No dataset or fine-tuned model is bundled. You must provide a labelled dataset and train the model before RoBERTa classification works.
- Search results are source-discovery leads, not fact-check verdicts. Search metadata may have incomplete dates or descriptions.
- The RAG assistant answers only from sources indexed in the current Streamlit session. Live search returns article metadata; to ground Q&A, fetch/index an article or paste/upload its text.
- GDELT search does not require an API key. To use NewsAPI, set `NEWSAPI_KEY` in PowerShell before launching Streamlit: `$env:NEWSAPI_KEY = 'your_key'`.
- URL ingestion may fail on paywalls, bot protection, JavaScript-only pages, or sites that disallow automated fetching. Paste the article text or upload a file instead.
- A model's REAL/FAKE classification is not independent fact-checking. Verify important claims with multiple credible sources.
- Training RoBERTa requires internet access to download the pretrained checkpoint and sufficient RAM/compute. GPU is recommended; CPU fine-tuning may be slow.
- Verify your dataset label mapping. This script maps REAL/TRUE/Genuine/0 to class 0 and FAKE/FALSE/1 to class 1. Do not use numeric labels without checking what they mean in your dataset.

## Windows quick start

1. Extract the ZIP to a folder such as `F:\FakeNews_Agent_RAG`.
2. Open PowerShell in that folder and run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Or install manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

## Prepare the dataset and train RoBERTa

The error `FileNotFoundError: data/news.csv` means the training CSV has not been created yet. This ZIP does not include copyrighted datasets. Download a labelled dataset and place its CSV files in `data`. For an ISOT-style dataset, place `True.csv` and `Fake.csv` in `data` after extracting the official dataset.

Combine the two files into the exact `data/news.csv` path required by training:

```powershell
python combine_news.py --real data/True.csv --fake data/Fake.csv --output data/news.csv
```

If the source files are somewhere else, pass their actual paths. The helper auto-detects common article text columns, or you can specify them with `--real-text-col` and `--fake-text-col`. It assigns REAL to the file passed as `--real` and FAKE to the file passed as `--fake`; verify those source labels before using it.

Then train:

```powershell
python train_roberta.py --csv data/news.csv --text-col text --label-col label --epochs 2 --batch-size 8 --max-length 256
```

Model saved to `models/roberta-fake-news/`; metrics saved to `reports/evaluation.json`.

For a different dataset schema, replace `text` and `label` with the actual column names. Check the source's label convention first.

## Ollama local explanations

Install Ollama from https://ollama.com/download, then:

```powershell
ollama pull llama3.2
ollama list
```

If needed, set another installed model in PowerShell before starting Streamlit:

```powershell
$env:OLLAMA_MODEL = 'llama3.2'
python -m streamlit run app.py
```

The app still supports retrieval-only fallback if Ollama is not running.

## Test

```powershell
python test_agents.py
python -m py_compile app.py train_roberta.py combine_news.py test_agents.py src\agents.py
```
