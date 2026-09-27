# EMNIST Character Recognition

A CNN trained on the EMNIST Balanced dataset. The API is deployed on [Render](https://render.com) and the Streamlit client is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) — training happens locally; only the trained artifacts are shipped.

## Project files

| File | Purpose |
|---|---|
| `training.py` | Loads the `.mat` dataset, trains the CNN, saves `modelo_emnist_balanced.keras` and `label_map.json` |
| `verification.py` | Plots a few real samples after preprocessing, to visually confirm orientation before committing to a full training run |
| `api.py` | FastAPI app: preprocesses incoming images and serves predictions from the trained model |
| `main.py` | Streamlit client (upload or draw a character, calls the API) |
| `test_preprocess.py` | Unit test for the API's image preprocessing |
| `label_map.json` | Maps model output indices to ASCII codes |

## Requirements

- Anaconda or Miniforge
- Python 3.12 (created automatically by `environment.yml`)
- The EMNIST Balanced MATLAB dataset

Create and activate the environment from Anaconda Prompt in the project directory:

```powershell
conda env create -f environment.yml
conda activate emnist-api
```

## 1. Train the model (local)

Download the MATLAB-format archive from the [official NIST EMNIST page](https://www.nist.gov/itl/products-and-services/emnist-dataset). Extract `emnist-balanced.mat` into a `matlab` folder in the project:

```text
AI-CNN/
  matlab/
    emnist-balanced.mat
```

Before running a full training pass, sanity-check the orientation fix on real data:

```powershell
python verification.py
```

This opens a grid of sample characters — confirm they're readable and match their titles. Then train:

```powershell
python training.py
```

This writes `modelo_emnist_balanced.keras` and `label_map.json` beside the scripts. The raw dataset stays out of Git; only the two trained artifacts are meant to travel further.

## 2. Run locally (optional, before deploying)

API:

```powershell
$env:API_KEY = "replace-with-a-long-random-secret"
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://127.0.0.1:8000/docs`; unauthenticated health check at `http://127.0.0.1:8000/health`.

Streamlit client, in a second terminal, with the API still running:

```powershell
$env:API_KEY = "replace-with-a-long-random-secret"
$env:API_URL = "http://127.0.0.1:8000"
streamlit run main.py
```

## 3. Tests

```powershell
pytest test_preprocess.py -v
```

Covers the image preprocessing in isolation — not a substitute for testing the deployed model with real drawings.

## 4. Deploy the API on Render

1. Push the repository to GitHub.
2. In the Render dashboard, **New +** → **Web Service**, and connect the repository.
3. Configure:

   | Setting | Value |
   |---|---|
   | Build command | `pip install -r requirements.txt` |
   | Start command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
   | Health check path | `/health` |

   Render assigns the port dynamically via `$PORT` — don't hardcode `8000` here.
4. Under **Environment**, add `API_KEY` with a long random value.
5. **Model artifacts:** Render's free-tier filesystem is ephemeral and only contains what's in the repository at build time. `modelo_emnist_balanced.keras` and `label_map.json` need to be committed to Git so they're present at deploy — **confirm this is how you did it**; if instead you're using a persistent disk or pulling the model from external storage at startup, that section needs different instructions.
6. Once deployed, the service is live at `https://YOUR-SERVICE-NAME.onrender.com`, already served over HTTPS with no extra configuration.

Free-tier services spin down after 15 minutes of inactivity and take 30–60 seconds to wake up on the next request — expect a slow first response after idle periods.

## 5. Deploy the Streamlit frontend on Streamlit Community Cloud

1. Push the same repository to GitHub (if not already).
2. At [share.streamlit.io](https://share.streamlit.io), **New app**, select the repo/branch, and set the main file path to `main.py`.
3. Under **Advanced settings → Secrets**, paste (root-level keys, no `[section]`, so they're readable via `os.getenv`):

   ```toml
   API_KEY = "same value as set on Render"
   API_URL = "https://YOUR-SERVICE-NAME.onrender.com"
   MIN_CONFIDENCE = "0.85"
   ```

4. Deploy. The app is served at `https://YOUR-APP-NAME.streamlit.app`, also HTTPS by default.

## Send a test request directly to the API

```bash
curl -X POST https://YOUR-SERVICE-NAME.onrender.com/predict \
  -H "X-API-Key: replace-with-a-long-random-secret" \
  -F "file=@character.png"
```

Response includes the predicted character, class index, and confidence. Accepted formats are those Pillow supports; uploads are limited to 10 MB.

## Security notes

- Both Render and Streamlit Community Cloud provision HTTPS automatically on their default subdomains — no reverse proxy or certificate setup needed for this deployment.
- Never commit `API_KEY` values, the raw dataset, or `.streamlit/secrets.toml` to Git.
- If a key is ever exposed (logs, a public commit, etc.), rotate it immediately in both Render's environment settings and Streamlit's secrets.
- A custom domain is optional; Render and Streamlit Cloud both issue TLS certificates for custom domains automatically if you add one later.

## Live Demo

Try it: **[[URL do teu Streamlit app](https://ocr-character-recognition-cnn-bvptx3xexfwzamrjfzg5o2.streamlit.app/)]**

> First request may take up to ~50s while the free-tier backend wakes up from inactivity — subsequent requests are fast.
