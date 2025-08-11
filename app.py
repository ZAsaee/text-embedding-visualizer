import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

# Lazy import for faster app boot
@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

@st.cache_data(show_spinner=False)
def compute_embeddings(texts, model_name: str):
    model = load_embedder(model_name)
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return emb

@st.cache_data(show_spinner=False)
def reduce_dims(emb, method: str, **kwargs):
    if method == "UMAP":
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=int(kwargs.get("n_neighbors", 15)),
            min_dist=float(kwargs.get("min_dist", 0.1)),
            metric="cosine",
            random_state=int(kwargs.get("random_state", 42)),
        )
        xy = reducer.fit_transform(emb)
    else:
        reducer = TSNE(
            n_components=2,
            perplexity=float(kwargs.get("perplexity", 30.0)),
            learning_rate="auto",
            init="pca",
            random_state=int(kwargs.get("random_state", 42)),
            metric="cosine",
            n_iter=1000,
        )
        xy = reducer.fit_transform(emb)
    return xy

st.set_page_config(page_title="Embedding Explorer", layout="wide")
st.title("🔎 Embedding Explorer")
st.write("Upload text, generate embeddings, and explore clusters in 2D (UMAP or t-SNE).")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Embedding model",
        ["sentence-transformers/all-MiniLM-L6-v2",
         "sentence-transformers/all-mpnet-base-v2",
         "sentence-transformers/paraphrase-MiniLM-L6-v2"],
        index=0,
        help="Local embeddings via sentence-transformers (no API keys)",
    )
    method = st.selectbox("Dimensionality reduction", ["UMAP", "t-SNE"], index=0)
    random_state = st.number_input("Random seed", 0, 10_000, 42)

    if method == "UMAP":
        n_neighbors = st.slider("UMAP: n_neighbors", 2, 200, 15)
        min_dist = st.slider("UMAP: min_dist", 0.0, 1.0, 0.10)
        perplexity = None
    else:
        perplexity = st.slider("t-SNE: perplexity", 5.0, 100.0, 30.0)
        n_neighbors = None
        min_dist = None

    st.subheader("Clustering")
    k = st.slider("KMeans: clusters (0 = off)", 0, 20, 6)

    st.subheader("Filter")
    query = st.text_input("Substring filter (optional)", value="")

st.subheader("Input Text")
st.write("Option A: upload a CSV with a **text** column. Option B: paste lines below (one document per line).")

up = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
default_lines = """Transformers are powerful models for sequence modeling.
Self-attention lets each token attend to every other token.
UMAP is a non-linear dimensionality reduction technique.
t-SNE is useful for visualizing high-dimensional data.
Embeddings map text to vectors in a semantic space.
KMeans partitions data into K clusters.
Sentence-Transformers provides high-quality sentence embeddings.
Clustering reveals structure in unlabeled data.
Interactive visualization helps communicate ML ideas.
This is a short example corpus to try the app quickly."""

txt = st.text_area("Or paste text (one per line)", height=160, value=default_lines)

def load_texts():
    if up is not None:
        df = pd.read_csv(up)
        if "text" not in df.columns:
            st.error("CSV must contain a 'text' column.")
            return None
        texts = df["text"].astype(str).tolist()
        return texts, df
    else:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if not lines:
            st.warning("Please upload a CSV or paste some text.")
            return None
        return lines, pd.DataFrame({"text": lines})

loaded = load_texts()
go = st.button("🚀 Embed & Visualize", type="primary", use_container_width=True)

if go and loaded is not None:
    texts, df_raw = loaded
    with st.spinner("Computing embeddings..."):
        emb = compute_embeddings(texts, model_name)

    with st.spinner(f"Reducing to 2D with {method}..."):
        xy = reduce_dims(
            emb, method,
            n_neighbors=n_neighbors, min_dist=min_dist,
            perplexity=perplexity, random_state=random_state
        )

    df = pd.DataFrame({"x": xy[:,0], "y": xy[:,1], "text": texts})

    # Clustering
    if k and k > 0 and len(df) >= k:
        km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
        labels = km.fit_predict(emb)
        df["cluster"] = labels.astype(int)
    else:
        df["cluster"] = -1

    # Filter
    if query:
        mask = df["text"].str.contains(query, case=False, na=False)
        df_plot = df[mask].copy()
        if df_plot.empty:
            st.info("No points match the filter; showing all data.")
            df_plot = df
    else:
        df_plot = df

    # Plot
    color = df_plot["cluster"].astype(str) if df_plot["cluster"].nunique() > 1 else None
    fig = px.scatter(
        df_plot, x="x", y="y",
        color=color,
        hover_data={"text": True, "x": False, "y": False, "cluster": True},
        template="plotly_white",
        height=620
    )
    fig.update_traces(marker=dict(size=9, opacity=0.85))
    fig.update_layout(legend_title_text="Cluster")

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # Downloads
    st.subheader("Downloads")
    emb_df = pd.DataFrame(emb)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
    out = pd.concat([df[["text","cluster","x","y"]], emb_df], axis=1)

    st.download_button(
        "Download embeddings CSV",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="embeddings_2d.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.subheader("How to read this")
    st.markdown(
        "- Each point is a text item mapped to a vector by the selected embedding model.\n"
        "- UMAP/t-SNE compress vectors into 2D so clusters become visible.\n"
        "- (Optional) KMeans assigns cluster IDs; use the filter to search by substring."
    )
else:
    st.info("Upload a CSV or paste text, then click **Embed & Visualize**.")

