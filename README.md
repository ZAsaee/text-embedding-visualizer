# Embedding Explorer

An interactive Streamlit app to **upload text**, generate **sentence embeddings** (sentence-transformers), reduce to 2D with **UMAP or t-SNE**, and **visualize clusters**. Optional **KMeans** clustering and substring filtering. No API keys required.

https://github.com/ (fork and deploy to Streamlit Cloud; add your live link here)

## 🔧 Quickstart (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy (Free)

- **Streamlit Community Cloud**: connect your GitHub repo, select `app.py`, deploy.
- **Hugging Face Spaces**: choose Streamlit template, point to your repo.

## 🧠 What it shows

- **Embeddings:** Local models like `all-MiniLM-L6-v2` map text → vectors.
- **2D projection:** UMAP or t-SNE reveals neighborhoods and topics.
- **Clustering:** Optional KMeans to label groups.
- **Exploration:** Hover to read points; filter by substring; download CSV with embeddings + 2D coords.

## 📄 CSV format

Upload a CSV with a column named `text`, for example:

```csv
text
"Transformers are powerful models for sequence modeling."
"Self-attention lets each token attend to every other token."
```

## 📸 Demo tips

- Record a 30–45s clip: upload → choose model → run → hover/zoom, show clusters.
- Comment hook for LinkedIn:
  > Built a tiny **Embedding Explorer**: paste a CSV, see clusters in seconds (UMAP/t‑SNE + KMeans). Local sentence-transformers—no API keys. Demo + code in first reply.

## 🔍 Notes

- UMAP is generally faster and preserves local structure; t‑SNE can be slower but often yields crisp clusters.
- KMeans on embeddings is a simple baseline; you can extend with HDBSCAN or topic modeling.
- This app runs fully local; swapping to cloud embeddings (Bedrock/OpenAI) is easy if you want to compare.

## 🧾 License

MIT
