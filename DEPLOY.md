# Deploy dashboard as shareable link (Streamlit Community Cloud)

## Option A: Streamlit Community Cloud (free, recommended)

1. Push this `tw_ai_stock/` folder to a GitHub repo (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app" -> select repo -> main file path: `app.py`.
4. In "Advanced settings" -> Secrets, add:
   ```
   FINMIND_TOKEN = "your_token_here"
   ```
5. Click Deploy. You'll get a public URL like `https://yourapp.streamlit.app`.

Notes:
- Streamlit Cloud reads `requirements.txt` automatically.
- The app caches data in `cache/`. For cloud, either commit the cache/ folder
  (parquet files) to the repo, or have the app fetch on first run (slow).
- Recommended: commit `cache/*.parquet` (excluding large raw) so dashboard loads fast.

## Option B: Hugging Face Spaces (Docker)

1. Create a Space at https://huggingface.co/spaces (SDK: Streamlit).
2. Upload all files including `requirements.txt`.
3. Add FINMIND_TOKEN as a Space secret.
4. Space auto-builds and gives a public URL.

## Option C: Local share via tunnel (temporary)

Install cloudflared (https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/),
then:
```
streamlit run app.py --server.port 8501
cloudflared tunnel --url http://localhost:8501
```
Gives a temporary public URL (lasts while process runs).

## Environment
- Python 3.12
- See requirements.txt
"""