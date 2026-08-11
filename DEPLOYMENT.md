# Deploying RentWise UK to Streamlit Community Cloud (Phase 10)

The app is a standard Streamlit project, so deployment is a few clicks once the
repo is on GitHub. No server admin needed.

## Prerequisites
1. The `Rentwise-UK` folder is pushed to a GitHub repo (see GITHUB_SETUP.md).
2. `requirements.txt` lists every dependency (it does).
3. The models exist. Streamlit Cloud runs `python run_pipeline.py`? No — instead,
   commit the trained models, OR add a first-run step. Simplest for a student demo:
   commit the small model files by removing `models/*.joblib` from `.gitignore`
   before pushing (a few MB), plus the small processed CSVs the dashboard reads
   (`data/processed/*.csv`) and the `reports/` figures, so the app loads the real
   trained model and results immediately. Rebuild them first with
   `python run_real_pipeline.py` (real data) or `python run_pipeline.py` (synthetic).

## Steps
1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **New app**.
3. Repository: `infokeshavkandel-bit/rentwise-uk`  ·  Branch: `main`.
4. Main file path: `app/streamlit_app.py`.
5. Click **Deploy**. First build installs the requirements (a few minutes).
6. You get a public URL like `https://rentwise-uk.streamlit.app` — put this link
   in your final report as the live artefact.

## Notes
- If the build times out installing heavy packages, pin lighter versions or
  remove `jupyter` from requirements (not needed at runtime).
- The app degrades gracefully: if a model file is missing it tells the user to run
  the pipeline, rather than crashing.
- Keep the "indicative, not financial advice" disclaimer visible (it is, in-app).
