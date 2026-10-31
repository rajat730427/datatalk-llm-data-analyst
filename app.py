import io
import re

import numpy as np
import ollama
import pandas as pd
import pdfplumber
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="DataTalk", page_icon="💬", layout="wide")

# ---------------------------------------------------------------- Design

PALETTE = ["#7C5CFF", "#22D3EE", "#F59E0B", "#10B981", "#F43F5E", "#60A5FA"]

pio.templates["datatalk"] = go.layout.Template(
    layout=go.Layout(
        colorway=PALETTE,
        font=dict(family="Inter, sans-serif", color="#E5E7F0"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.15)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.15)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(font=dict(size=18)),
        margin=dict(l=40, r=20, t=60, b=40),
        hoverlabel=dict(bgcolor="#131A2E", font_color="#E5E7F0"),
    )
)
pio.templates.default = "plotly_dark+datatalk"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b1450 0%, #0B1020 55%) fixed; color: #E5E7F0; }

#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1200px; }

/* sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #121833 0%, #0B1020 100%); border-right: 1px solid #26304D; }
.brand { font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.2rem; }
.brand span { background: linear-gradient(90deg, #7C5CFF, #22D3EE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.tag { color: #8B93B0; font-size: 0.82rem; margin-bottom: 1rem; }
[data-testid="stFileUploader"] section { background: #131A2E; border: 1.5px dashed #3A4670; border-radius: 14px; }
[data-testid="stFileUploader"] section:hover { border-color: #7C5CFF; }
.stTextInput input { background: #131A2E; border: 1px solid #26304D; border-radius: 10px; color: #E5E7F0; }

/* hero */
.hero { padding: 1.6rem 1.8rem; border-radius: 20px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(124,92,255,0.25), rgba(34,211,238,0.10));
        border: 1px solid #2E3A66; }
.hero-badge { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; color: #22D3EE;
              background: rgba(34,211,238,0.12); padding: 0.25rem 0.7rem; border-radius: 999px; margin-bottom: 0.7rem; }
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em; color: #FFFFFF; }
.hero p { margin: 0.4rem 0 0 0; color: #AEB6D4; font-size: 1.02rem; }

/* empty state */
.empty { text-align: center; padding: 3rem 1rem; border: 1.5px dashed #2E3A66; border-radius: 20px; color: #AEB6D4; background: rgba(19,26,46,0.6); }
.empty .big { font-size: 2.6rem; }
.empty .title { font-size: 1.2rem; font-weight: 700; color: #fff; margin: 0.4rem 0; }

/* KPI cards */
[data-testid="stMetric"] { background: #131A2E; border: 1px solid #26304D; border-radius: 16px; padding: 0.9rem 1.1rem; }
[data-testid="stMetricLabel"] { color: #8B93B0; }
[data-testid="stMetricValue"] { font-weight: 800; color: #fff; }

/* expanders, buttons */
[data-testid="stExpander"] { background: #131A2E; border: 1px solid #26304D; border-radius: 14px; }
.stButton button { background: #131A2E; border: 1px solid #2E3A66; border-radius: 12px; color: #E5E7F0; transition: all .15s; }
.stButton button:hover { border-color: #7C5CFF; color: #fff; transform: translateY(-1px); }

/* chat */
[data-testid="stChatMessage"] { background: #131A2E; border: 1px solid #26304D; border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: 0.8rem; }
[data-testid="stChatInput"] { border-radius: 14px; }
[data-testid="stChatInput"] textarea { color: #E5E7F0; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------- LLM helpers


def ask_llm(model: str, system: str, user: str) -> str:
    r = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.1},
    )
    return r["message"]["content"]


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


CODE_SYSTEM = """You are an expert Python data analyst.
A pandas DataFrame named `df` is ALREADY loaded. Write Python code to answer the user's question.
Rules:
- Reply with ONE ```python code block and nothing else.
- Available names: df, pd, np, px (plotly.express), go (plotly.graph_objects).
- Never read/write files, never use input(), never call fig.show() or plt.show().
- ALWAYS store the final answer in a variable named `result` (aggregate with groupby first; never leave it None).
- If a chart helps, build a Plotly figure FROM `result` (not from the raw df) and store it in a variable named `fig`.
- Use only column names that exist in the schema.
- Date columns may be plain strings: convert with pd.to_datetime first. For monthly grouping use df['date'].dt.strftime('%Y-%m'). NEVER use .dt.to_period() or Period objects."""

INSIGHT_SYSTEM = """You are a data analyst. You are given the user's question and the result.
Write a 2-3 line insight in English.
If "Computed facts" are provided, use those numbers exactly. Never do arithmetic yourself.
Name the highest and lowest items and mention the gap between them.
Do not add currency symbols like $ to numbers.
Use only the numbers present in the result. Do not invent anything."""

TEXT_SYSTEM = """You are a document assistant. Answer only from the document excerpts provided.
If the answer is not in the document, clearly say it was not found.
Reply in English."""

# ---------------------------------------------------------------- File loading


def to_num(col: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(col)
    except (ValueError, TypeError):
        return col


@st.cache_data(show_spinner=False)
def load_file(name: str, data: bytes):
    """Returns (df | None, text | None)."""
    lower = name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data)), None
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data)), None
    if lower.endswith(".pdf"):
        tables, texts = [], []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
                for t in page.extract_tables():
                    if t and len(t) >= 2:
                        tables.append(t)
        df = None
        if tables:
            header = [str(h) for h in tables[0][0]]
            rows = [r for t in tables if len(t[0]) == len(header) for r in t]
            rows = [r for r in rows if [str(c) for c in r] != header]
            df = pd.DataFrame(rows, columns=header)
            df = df.apply(to_num)
        return df, "\n".join(texts)
    raise ValueError("Unsupported file type")


def schema_of(df: pd.DataFrame) -> str:
    return (
        f"Shape: {df.shape}\n"
        f"Columns & dtypes:\n{df.dtypes.to_string()}\n\n"
        f"First 5 rows:\n{df.head(5).to_string()}"
    )


# ---------------------------------------------------------------- Execution


def run_code(code: str, df: pd.DataFrame):
    ns = {"df": df.copy(), "pd": pd, "np": np, "px": px, "go": go,
          "result": None, "fig": None}
    exec(code, ns)  # NOTE: runs LLM-written code on your machine
    return make_safe(ns.get("result"), ns.get("fig"))


def make_safe(result, fig):
    """Convert pandas Period values (not JSON serializable) to strings."""
    if isinstance(result, pd.Series):
        result = result.copy()
        if isinstance(result.index, pd.PeriodIndex):
            result.index = result.index.astype(str)
        if isinstance(result.dtype, pd.PeriodDtype):
            result = result.astype(str)
    elif isinstance(result, pd.DataFrame):
        result = result.copy()
        if isinstance(result.index, pd.PeriodIndex):
            result.index = result.index.astype(str)
        for c in result.columns:
            if isinstance(result[c].dtype, pd.PeriodDtype):
                result[c] = result[c].astype(str)
    if fig is not None and hasattr(fig, "data"):
        for trace in fig.data:
            for attr in ("x", "y"):
                vals = getattr(trace, attr, None)
                if vals is not None and len(vals) and any(isinstance(v, pd.Period) for v in vals):
                    setattr(trace, attr, [str(v) for v in vals])
    return result, fig


def analyse(model: str, df: pd.DataFrame, question: str, retries: int = 2):
    prompt = f"Schema:\n{schema_of(df)}\n\nQuestion: {question}"
    last_err = None
    for _ in range(retries + 1):
        if last_err:
            prompt += f"\n\nThe previous code failed with this error:\n{last_err}\nFix the code and try again."
        code = extract_code(ask_llm(model, CODE_SYSTEM, prompt))
        try:
            result, fig = run_code(code, df)
            if result is None:
                raise ValueError("Variable `result` was not set. Store the aggregated table in `result` and build `fig` from it.")
            return code, result, fig, None
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
    return code, None, None, last_err


def result_to_text(result) -> str:
    if result is None:
        return "(no result)"
    if isinstance(result, pd.Series):
        result = result.reset_index()
    if isinstance(result, pd.DataFrame):
        text = result.head(20).to_string()[:3000]
        num_cols = list(result.select_dtypes("number").columns)
        cat_cols = [c for c in result.columns if c not in num_cols]
        if len(result) >= 2 and num_cols and cat_cols:
            n, c = num_cols[-1], cat_cols[0]
            hi = result.loc[result[n].idxmax()]
            lo = result.loc[result[n].idxmin()]
            text += (
                "\n\nComputed facts (use these exactly, do not recalculate):\n"
                f"Highest: {hi[c]} = {hi[n]:,.2f}\n"
                f"Lowest: {lo[c]} = {lo[n]:,.2f}\n"
                f"Gap (highest - lowest): {hi[n] - lo[n]:,.2f}\n"
                f"Total: {result[n].sum():,.2f}"
            )
        return text
    return str(result)[:3000]


# ---------------------------------------------------------------- Text (PDF) Q&A


def top_chunks(text: str, question: str, size: int = 1500, k: int = 4) -> str:
    chunks = [text[i:i + size] for i in range(0, len(text), size)]
    q = set(re.findall(r"\w+", question.lower()))
    scored = sorted(chunks, key=lambda c: len(q & set(re.findall(r"\w+", c.lower()))), reverse=True)
    return "\n---\n".join(scored[:k])


# ---------------------------------------------------------------- UI helpers

AVATARS = {"user": "🧑‍💻", "assistant": "✨"}


def render_message(m: dict):
    with st.chat_message(m["role"], avatar=AVATARS.get(m["role"])):
        st.markdown(m["content"].replace("$", "\\$"))
        if m.get("code"):
            with st.expander("Generated code"):
                st.code(m["code"], language="python")
        res = m.get("result")
        if isinstance(res, (pd.DataFrame, pd.Series)):
            st.dataframe(res, use_container_width=True)
        elif res is not None:
            st.write(res)
        fig = m.get("fig")
        if fig is not None:
            if hasattr(fig, "savefig"):  # matplotlib
                st.pyplot(fig)
            else:
                try:
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not draw the chart: {e}")


def suggestions_for(df: pd.DataFrame) -> list:
    nums = list(df.select_dtypes("number").columns)
    cats = [c for c in df.columns if c not in nums and df[c].nunique() <= 20]
    out = ["Show a summary of missing values per column"]
    if nums and cats:
        out.insert(0, f"Show total {nums[-1]} by {cats[0]} as a bar chart")
    if nums:
        out.insert(1, f"Show the distribution of {nums[-1]} as a histogram")
    return out[:3]


# ---------------------------------------------------------------- App

with st.sidebar:
    st.markdown('<div class="brand">💬 <span>DataTalk</span></div>'
                '<div class="tag">Chat with your data. Runs 100% locally.</div>',
                unsafe_allow_html=True)
    model = st.text_input("Ollama model", value="qwen2.5-coder:7b")
    uploaded = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "pdf"])
    st.caption("Run `ollama pull qwen2.5-coder:7b` first.")

st.markdown(
    '<div class="hero"><div class="hero-badge">LOCAL AI · PRIVATE</div>'
    '<h1>Ask your data anything</h1>'
    '<p>Upload a CSV or Excel file, ask in plain English, and get code, charts and insights.</p></div>',
    unsafe_allow_html=True,
)

if not uploaded:
    st.markdown(
        '<div class="empty"><div class="big">📂</div>'
        '<div class="title">No file yet</div>'
        'Upload a CSV, Excel or PDF from the sidebar to get started.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

file_key = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    st.session_state.file_key = file_key
    st.session_state.messages = []

try:
    df, text = load_file(uploaded.name, uploaded.getvalue())
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the file: {e}")
    st.stop()

modes = []
if df is not None:
    modes.append("Data analysis (code + chart)")
if text and text.strip():
    modes.append("Document Q&A (text)")
if not modes:
    st.error("No table or text found in this file. A scanned PDF would need OCR.")
    st.stop()

mode = st.sidebar.radio("Mode", modes) if len(modes) > 1 else modes[0]

if mode.startswith("Data"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric columns", len(df.select_dtypes("number").columns))
    c4.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    with st.expander("Data preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

for m in st.session_state.messages:
    render_message(m)

if mode.startswith("Data") and not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    for col, s in zip(cols, suggestions_for(df)):
        if col.button(s, use_container_width=True):
            st.session_state.pending = s

question = st.chat_input("Ask a question about your data...")
if not question and st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    try:
        if mode.startswith("Data"):
            with st.spinner("Writing and running code..."):
                code, result, fig, err = analyse(model, df, question)
            if err:
                reply = {"role": "assistant", "code": code,
                         "content": f"❌ The code failed to run:\n\n`{err}`"}
            else:
                with st.spinner("Writing insight..."):
                    insight = ask_llm(
                        model, INSIGHT_SYSTEM,
                        f"Question: {question}\n\nResult:\n{result_to_text(result)}",
                    )
                reply = {"role": "assistant", "content": insight,
                         "code": code, "result": result, "fig": fig}
        else:
            with st.spinner("Reading the document..."):
                ctx = top_chunks(text, question)
                answer = ask_llm(model, TEXT_SYSTEM,
                                 f"Document excerpts:\n{ctx}\n\nQuestion: {question}")
            reply = {"role": "assistant", "content": answer}
    except Exception as e:  # noqa: BLE001
        reply = {"role": "assistant",
                 "content": f"❌ Error: `{e}`\n\nCheck that Ollama is running (`ollama serve`)."}

    st.session_state.messages.append(reply)
    render_message(reply)