# %%
# %% [1] Imports and configuration
import socket
from pathlib import Path
import gradio as gr
import subprocess
import shlex
import os, re, json, hashlib, base64
from math import sqrt
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr
from openai import OpenAI
from curator import load_topics


api_key = os.environ["OPENAI_API_KEY"] 
# OpenAI client – DO NOT pass project="..." here
client = OpenAI()   # reads OPENAI_API_KEY from environment

try:
    import fitz  # PyMuPDF (optional for PDFs)
except Exception:
    fitz = None

# ---------- Config ----------


# Text embedding model for RAG
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Main language model for text Q&A
DEFAULT_LLM_MODEL = "gpt-5.2"# text + vision capable

# Vision model for image-based questions (same model can handle images)
VISION_MODEL = "gpt-5.2"

# Retrieval settings
TOP_K_DEFAULT = 6
TEMP_DEFAULT  = 0.20
CHUNK_MIN_LEN = 30

# Cache directory
CACHE_DIR = Path(".rag_cache")
CACHE_DIR.mkdir(exist_ok=True)


# ---------- Knowledge Bases ----------
CUR = load_topics("Knowledge_Base/topics.json")
KNOWLEDGE_BASES = CUR.knowledge_bases
DEFAULT_TOPIC = CUR.default_topic
#debug cell for openAi model 

from openai import OpenAIError

try:
    resp = client.models.list()
    print("✅ OpenAI client works, models available:", len(resp.data))
except OpenAIError as e:
    print("❌ OpenAI error:", e)

# %% [2] Helper functions: PDF reading, hashing, embedding utils

def pull_model(model: str):
    """Attempt 'ollama pull <model>' but fail silently."""
    try:
        subprocess.run(shlex.split(f"ollama pull {model}"), check=True)
    except Exception:
        pass

def _hash_for_cache(file_path: Path, embedding_model: str) -> str:
    """Compute hash for caching embeddings."""
    m = hashlib.md5()
    m.update(embedding_model.encode("utf-8"))

    if file_path.exists():
        st = file_path.stat()
        m.update(str(file_path.resolve()).encode("utf-8"))
        m.update(str(st.st_size).encode("utf-8"))
        m.update(str(int(st.st_mtime)).encode("utf-8"))

    return m.hexdigest()

def _extract_paragraphs_from_pdf(pdf_path: Path) -> List[str]:
    if fitz is None:
        raise RuntimeError("Install PyMuPDF to read PDFs.")

    doc = fitz.open(str(pdf_path))
    paras = []
    for page in doc:
        text = page.get_text("text") or ""
        if not text.strip():
            continue

        text = text.replace("\r\n", "\n")
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\n(?!\n)', ' ', text)

        for p in re.split(r'\n{2,}', text):
            p = re.sub(r'[ \t]+', ' ', p).strip()
            if len(p) >= CHUNK_MIN_LEN:
                paras.append(p)

    doc.close()
    return paras

def load_paragraphs(path: Path) -> List[str]:
    """Load paragraphs from either txt or PDF."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        return _extract_paragraphs_from_pdf(path)

    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return [c for c in chunks if len(c) >= CHUNK_MIN_LEN]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0
# %% [3] Embeddings, vector DB, STRICT RAG answer function

from typing import List, Tuple

# ---------- Similarity helper ----------

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ---------- Vector DB ----------

class MiniVectorDB:
    def __init__(self):
        # list of (chunk_text, embedding_vector)
        self.rows: List[Tuple[str, list]] = []

    def add(self, chunk: str, emb: list):
        self.rows.append((chunk, emb))

    def retrieve(self, q_emb: list, top_k: int):
        scored = [
            (chunk, cosine_similarity(q_emb, emb))
            for chunk, emb in self.rows
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ---------- Embeddings via OpenAI ----------

def embed_text(text: str, model: str = DEFAULT_EMBEDDING_MODEL) -> list:
    """
    Get an embedding vector for a piece of text using OpenAI.
    """
    resp = client.embeddings.create(
        model=model,
        input=text,
    )
    return resp.data[0].embedding


def build_db(paragraphs: List[str], model: str, progress=None) -> MiniVectorDB:
    """
    Build a MiniVectorDB from a list of paragraphs.
    `model` is passed in but we still call embed_text() with DEFAULT_EMBEDDING_MODEL.
    """
    db = MiniVectorDB()
    n = len(paragraphs)
    for i, p in enumerate(paragraphs, start=1):
        emb = embed_text(p, model=DEFAULT_EMBEDDING_MODEL)
        db.add(p, emb)
        if progress and (i % max(1, n // 20) == 0 or i == n):
            progress((i / n, f"Embedding {i}/{n}"))
    return db


# ---------- STRICT RAG answer via OpenAI ----------

# If similarity of best match is below this, we will NOT call the LLM at all.
# Instead, we immediately return "I don't know..." to avoid hallucinations.
STRICT_SIM_THRESHOLD = 0.2   # tweakable: try 0.25–0.35

def answer_with_context(question: str, db, temp: float = TEMP_DEFAULT):
    hits = _search_db(db, question, k=TOP_K_DEFAULT)
    context_text = "\n\n".join(chunk for chunk, _ in hits)

    top_score = hits[0][1] if hits else 0.0
    notes_supported = (len(hits) > 0) and (top_score >= MIN_SUPPORT_SCORE)

    if notes_supported:
        system_msg = (
            "You are a careful OSU ECEN teaching assistant.\n"
            "Answer using ONLY the course notes provided in the context.\n"
            "The provided context IS relevant to the question.\n"
            "DO NOT say 'Not found in notes' or anything similar.\n\n"
            "• Do NOT say phrases like 'listed in the notes', 'in the notes', or 'according to the notes'. Just present the answer.\n\n"
            "=== EQUATION FORMAT RULES ===\n"
            "• ANY time you write a formula or equation, you MUST format it in LaTeX.\n"
            "• Use DISPLAY MATH only, like this:\n"
            "    \\[ A_{v,\\mathrm{dB}} = 20 \\log_{10}(A_v) \\]\n"
            "• Do NOT use backticks or code blocks for math.\n"
            "• Keep derivation steps short, numbered, and clean.\n"
        )
    else:
        system_msg = (
            "You are a careful OSU ECEN teaching assistant.\n"
            "You must answer using ONLY the course notes provided in the context.\n"
            "If the notes do NOT contain enough information, FIRST say:\n"
            "=== EQUATION FORMAT RULES ===\n"
            "• ANY time you write a formula or equation, you MUST format it in LaTeX.\n"
            "• Use DISPLAY MATH only, like this:\n"
            "    \\[ A_{v,\\mathrm{dB}} = 20 \\log_{10}(A_v) \\]\n"
            "• Do NOT use backticks or code blocks for math.\n"
            "• Keep derivation steps short, numbered, and clean.\n"
        )

    user_msg = (
        "Use ONLY the following course notes as your primary reference.\n\n"
        "=== COURSE NOTES START ===\n"
        + context_text +
        "\n=== COURSE NOTES END ===\n\n"
        "Student question:\n"
        + question +
        "\n"
    )

    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temp,
    )

    answer = resp.choices[0].message.content or ""

    # Extra safety: if notes_supported=True but the model still prints the warning, strip it.
    if notes_supported:
        answer = re.sub(
            r"^\s*Not found in notes\s*—.*?\.\s*",
            "",
            answer,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()

    return answer, hits

# %% [4] State container with topic selection support

class State:
    def __init__(self):
        self.topic = DEFAULT_TOPIC
        self.file = KNOWLEDGE_BASES[self.topic]
        self.paragraphs: List[str] = []
        self.db: Optional[MiniVectorDB] = None
        self.cache_key = ""

    def cache_file(self):
        return CACHE_DIR / f"{self.cache_key}.json"

    def try_load_cache(self):
        f = self.cache_file()
        if not f.exists():
            return False
        try:
            data = json.loads(f.read_text())
            db = MiniVectorDB()
            for r in data["rows"]:
                db.add(r["chunk"], r["embedding"])
            self.db = db
            return True
        except:
            return False

    def save_cache(self):
        if not self.db:
            return
        rows = [{"chunk": c, "embedding": emb} for c, emb in self.db.rows]
        self.cache_file().write_text(json.dumps({"rows": rows}), encoding="utf-8")


STATE = State()

# %% [5] Callbacks (Good answers + NO outside-KB leakage via evidence verification)

import re
import gradio as gr
from openai import OpenAI

client = OpenAI()

# -------------------- Tunables --------------------

# If retrieval is weak, we refuse instead of letting the LLM guess.
MIN_SUPPORT_SCORE_LONG  = 0.34   # normal questions
MIN_SUPPORT_SCORE_SHORT = 0.22   # short queries like "who is soren"

# Hybrid retrieval weights (helps short/heading queries)
LEX_BONUS_WEIGHT    = 0.45
EXACT_PHRASE_BONUS  = 0.20

# If True, allow vision fallback (outside-notes). If you want "notes only", set False.
ALLOW_VISION_FALLBACK = False

NO_ANSWER_MESSAGE = "I don't have enough information in the provided notes to answer that."

_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were",
    "who","what","when","where","why","how","do","does","did","tell","me","about","please","can", "use"

}

# -------------------- Utilities --------------------

def _content_to_str(content) -> str:
    """Normalize OpenAI content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
            elif isinstance(p, str):
                parts.append(p)
            else:
                parts.append(str(p))
        return "".join(parts)
    if isinstance(content, dict) and "text" in content:
        return str(content["text"])
    return str(content)

def normalize_latex_for_gradio(text: str) -> str:
    """Normalize LaTeX to $$...$$ and remove newline edge cases."""
    text = text or ""
    text = re.sub(r"\\\[(.*?)\\\]", r"$$ \1 $$", text, flags=re.DOTALL)
    text = re.sub(r"\$\$\s*\n+", "$$ ", text)
    text = re.sub(r"\n+\s*\$\$", " $$", text)
    return text

def _to_messages_history(history):
    """
    Ensure history is messages format:
      [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    Also prevents [{text,type}] artifacts by flattening content to strings.
    """
    if history is None:
        return []

    out = []

    # Old tuples format
    if isinstance(history, list) and history and isinstance(history[0], (list, tuple)) and len(history[0]) == 2:
        for u, a in history:
            u_txt = _content_to_str(u).strip()
            a_txt = _content_to_str(a).strip()
            if u_txt:
                out.append({"role": "user", "content": u_txt})
            if a_txt:
                out.append({"role": "assistant", "content": a_txt})
        return out

    # Messages format
    if isinstance(history, list):
        for m in history:
            if isinstance(m, dict) and "role" in m and "content" in m:
                out.append({"role": str(m["role"]), "content": _content_to_str(m["content"])})
        return out

    return []

def _tokenize(text: str):
    return re.findall(r"[a-z0-9]+", (text or "").lower())

def _keyword_tokens(query: str):
    toks = _tokenize(query)
    out = []
    seen = set()
    for t in toks:
        if t in _STOPWORDS:
            continue
        if len(t) >= 3 or t == "ai":
            if t not in seen:
                out.append(t)
                seen.add(t)
    return out


def _dynamic_threshold(question: str) -> float:
    n = len(_keyword_tokens(question))  # count only meaningful words
    return MIN_SUPPORT_SCORE_SHORT if n <= 4 else MIN_SUPPORT_SCORE_LONG


# -------------------- Retrieval --------------------

def embed(text: str, model: str = DEFAULT_EMBEDDING_MODEL):
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def _search_db(db, query: str, k: int = TOP_K_DEFAULT):
    """
    Hybrid scoring:
      total = cosine_sim + LEX_BONUS_WEIGHT*lex_overlap + EXACT_PHRASE_BONUS*phrase_hit
    Returns list of (chunk, total_score).
    """
    if db is None or not getattr(db, "rows", None):
        return []

    q_lower = (query or "").strip().lower()
    q_tokens = _keyword_tokens(query)
    q_emb = embed(query, DEFAULT_EMBEDDING_MODEL)

    scored = []
    for chunk, emb in db.rows:
        c_lower = (chunk or "").lower()

        sim = cosine_similarity(q_emb, emb)

        if q_tokens:
            hits = sum(1 for t in q_tokens if t in c_lower)
            lex = hits / len(q_tokens)
        else:
            lex = 0.0

        phrase_hit = 1.0 if (q_lower and len(q_lower) >= 4 and q_lower in c_lower) else 0.0

        total = sim + (LEX_BONUS_WEIGHT * lex) + (EXACT_PHRASE_BONUS * phrase_hit)
        scored.append((chunk, total))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]

# -------------------- Evidence-verified answering --------------------

def _parse_final_and_evidence(raw: str):
    """
    Expect format:

    FINAL:
    ...

    EVIDENCE:
    "quote1"
    "quote2"

    Returns (final_text, evidence_quotes_list)
    """
    raw = (raw or "").strip()

    # If the model didn't follow format, treat everything as final (will fail verification)
    if "EVIDENCE:" not in raw:
        return raw, []

    # Split on EVIDENCE:
    parts = raw.split("EVIDENCE:", 1)
    left = parts[0].strip()
    right = parts[1].strip()

    # Remove leading "FINAL:" if present
    left = re.sub(r"^\s*FINAL:\s*", "", left, flags=re.IGNORECASE).strip()

    # Evidence quotes must be in double quotes
    quotes = re.findall(r'"([^"]{10,})"', right)
    return left, quotes

def _evidence_is_valid(quotes, context_text: str) -> bool:
    """
    Verify quotes are exact substrings of the retrieved context.
    Require at least one non-trivial quote.
    """
    if not quotes:
        return False
    ctx = context_text or ""
    # Require at least one quote length >= 20 to avoid trivial "diode" etc.
    if max(len(q) for q in quotes) < 20:
        return False
    return all(q in ctx for q in quotes)

def answer_with_context(question: str, db, temp: float = TEMP_DEFAULT):
    hits = _search_db(db, question, k=TOP_K_DEFAULT)
    top_score = hits[0][1] if hits else 0.0
    thr = _dynamic_threshold(question)

    # If retrieval is too weak, refuse (prevents guessing/outside info).
    if (not hits) or (top_score < thr):
        return "", hits, False

    context_text = "\n\n".join(chunk for chunk, _ in hits)

    system_msg = (
        "You are a careful OSU ECEN teaching assistant.\n"
        "You MUST use ONLY the provided course notes context.\n"
        "Do NOT use outside knowledge. Do NOT invent modules/weeks/details.\n\n"
        "You must respond in EXACTLY this format:\n"
        "FINAL:\n"
        "<your answer in a helpful style>\n\n"
        "EVIDENCE:\n"
        "\"<verbatim quote from the context>\"\n"
        "\"<another verbatim quote from the context if needed>\"\n\n"
        "Rules:\n"
        "- Every factual claim in FINAL must be supported by the EVIDENCE quotes.\n"
        "- EVIDENCE quotes must be copied EXACTLY from the provided context.\n"
        "- If the context does not contain enough information, output ONLY:\n"
        "  FINAL:\n"
        "  I don't have enough information in the provided notes to answer that.\n"
        "  EVIDENCE:\n"
        "  \"\"\n\n"
        "=== EQUATION FORMAT RULES ===\n"
        "- Any equation must be display math using $$ ... $$ on ONE LINE.\n"
        "- Do NOT use \\[ ... \\] or \\( ... \\).\n"
        "- Do NOT use backticks/code blocks for math.\n"
    )

    user_msg = (
        "=== COURSE NOTES START ===\n"
        + context_text +
        "\n=== COURSE NOTES END ===\n\n"
        "Student question:\n"
        + question
    )

    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temp,
    )

    raw = _content_to_str(resp.choices[0].message.content).strip()
    raw = normalize_latex_for_gradio(raw)

    final_text, quotes = _parse_final_and_evidence(raw)

    # If model says it lacks info, allow it (no evidence needed).
    if final_text.strip() == NO_ANSWER_MESSAGE:
        return final_text, hits, True

    # Evidence verification gate: if quotes aren't valid, refuse.
    if not _evidence_is_valid(quotes, context_text):
        return "", hits, False

    # Clean final output (no banners/disclaimers)
    return final_text.strip(), hits, True

# -------------------- Topic loading (unchanged) --------------------

def load_topic(topic, progress=gr.Progress(track_tqdm=False)):
    STATE.topic = topic
    STATE.file = KNOWLEDGE_BASES[topic]
    STATE.cache_key = _hash_for_cache(STATE.file, DEFAULT_EMBEDDING_MODEL)

    try:
        STATE.paragraphs = load_paragraphs(STATE.file)
    except Exception as e:
        return f"❌ Error loading file: {e}"

    if STATE.try_load_cache():
        return f"✅ Loaded {len(STATE.paragraphs)} chunks from cache ({STATE.file.name})."

    STATE.db = build_db(STATE.paragraphs, DEFAULT_EMBEDDING_MODEL, progress)
    STATE.save_cache()
    return f"✅ Indexed {len(STATE.paragraphs)} chunks from {STATE.file.name}."

# -------------------- Main callback --------------------

def ask_question(chat_history, user_msg, show_sources, problem_image=None):
    chat_history = _to_messages_history(chat_history)

    if not user_msg or not user_msg.strip():
        return chat_history, "", gr.update(visible=False, value="")

    if STATE.db is None:
        return chat_history, "", gr.update(visible=True, value="⚠ Load a knowledge base first.")

    user_msg = user_msg.strip()
    chat_history.append({"role": "user", "content": user_msg})

    answer, ctx = "", []
    supported = False

    answer, ctx, supported = answer_with_context(user_msg, STATE.db)

    used_vision = False
    if (not supported) and problem_image and ALLOW_VISION_FALLBACK:
        answer = answer_with_image(problem_image, user_msg)  # Cell 5b
        answer = normalize_latex_for_gradio(answer)
        used_vision = True
        supported = True  # but sources will show vision note

    if (not supported) and (not used_vision):
        answer = NO_ANSWER_MESSAGE

    chat_history.append({"role": "assistant", "content": answer})

    if show_sources:
        if used_vision:
            src_md = "_Vision fallback used (outside notes). No note chunks were used._"
        else:
            src_md = "\n\n".join(
                f"**{i+1}.** {chunk}\n_score={score:.3f}_"
                for i, (chunk, score) in enumerate(ctx)
            ) if ctx else "_No sources retrieved._"
        return chat_history, "", gr.update(visible=True, value=src_md)

    return chat_history, "", gr.update(visible=False, value="")

def clear_chat():
    return [], "", gr.update(visible=False, value="")

def generate_quiz(num_q):
    if not STATE.paragraphs:
        return "⚠ Load a topic first."

    context = "\n\n".join(STATE.paragraphs[:min(12, len(STATE.paragraphs))])
    prompt = (
        "Using ONLY this context, create "
        + str(int(num_q))
        + " conceptual quiz questions.\n\nContext:\n"
        + context
    )

    resp = client.chat.completions.create(
        model=DEFAULT_LLM_MODEL,
        messages=[
            {"role": "system", "content": "Write clear ECEN quiz questions using ONLY the provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return _content_to_str(resp.choices[0].message.content).strip()

# %% [5b] Vision helper – explain an uploaded image inside the chat (OpenAI)
# No "Not found in notes" sentence. Still enforces $$...$$ math.

import base64
import re

def answer_with_image(image_path: str, question: str) -> str:
    """
    Use the OpenAI vision model to answer a question about an uploaded image.
    Used ONLY after RAG fails to find an answer in notes.
    """
    if not image_path:
        return "⚠ No image received. Please upload a picture."

    if not (question or "").strip():
        question = (
            "Explain this problem step by step as an ECEN TA. "
            "Identify what the problem is asking, extract all given information, "
            "and show the solution process clearly."
        )

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        b64 = base64.b64encode(img_bytes).decode("utf-8")

        lower = image_path.lower()
        if lower.endswith(".jpg") or lower.endswith(".jpeg"):
            mime = "image/jpeg"
        else:
            mime = "image/png"

        image_url = f"data:{mime};base64,{b64}"

        system_msg = (
            "You are an ECEN 3314 teaching assistant.\n"
            "Solve the student's question based on the image.\n\n"
            "=== EQUATION FORMAT RULES ===\n"
            "• ANY time you write a formula or equation, you MUST format it in LaTeX.\n"
            "• Use DISPLAY MATH only, wrapped like this on ONE LINE:\n"
            "    $$ V_{out} = A_v (V^+ - V^-) $$\n"
            "• Do NOT use \\[ ... \\] or \\( ... \\).\n"
            "• Do NOT use backticks or code blocks for math.\n"
            "• Keep derivation steps short, numbered, and clean.\n"
        )

        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            temperature=0.1,
        )

        out = (resp.choices[0].message.content or "").strip()

        # Normalize \[...\] -> $$...$$ and remove newline edge cases for rendering
        out = re.sub(r"\\\[(.*?)\\\]", r"$$ \1 $$", out, flags=re.DOTALL)
        out = re.sub(r"\$\$\s*\n+", "$$ ", out)
        out = re.sub(r"\n+\s*\$\$", " $$", out)

        # Remove any accidental "Not found in notes..." line if the model includes it anyway
        out = re.sub(r"^\s*not found in notes[^\n]*\n+", "", out, flags=re.IGNORECASE).strip()

        return out

    except Exception as e:
        return f"❌ Vision model error: {e}"
# %% [6] UI definition – OSU theme + logo + multi-KB + quiz + controls UNDER question + larger chat

import gradio as gr

OSU_THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="amber")

def build_ui():
    with gr.Blocks(title="OSU Virtual TA") as demo:

        # ---- OSU Dark Theme + Mobile Fixes ----
        gr.HTML("""
        <style>
            body, .gr-blocks { background-color: #111 !important; color: #fff !important; }

            .gr-chatbot { background-color: #101218 !important; border-radius: 10px !important; }
            .gr-chatbot .message.user { background: #ff7300 !important; color: #000 !important; }
            .gr-chatbot .message.bot  { background: #22252d !important; color: #fff !important; }

            .gr-button-primary {
                background-color: #ff7300 !important;
                color: #000 !important;
                border-color: #ff7300 !important;
                font-weight: 600 !important;
            }

            .gr-textbox textarea {
                background-color: #20232b !important;
                color: #fff !important;
                border-radius: 10px !important;
            }

            /* Make the chat use more vertical space on desktop */
            .gr-chatbot { min-height: 680px !important; }

            /* ---- Title text wrapping fix ---- */
            #osu_title_block {
                word-break: normal !important;
                overflow-wrap: normal !important;
                white-space: normal !important;
            }

            /* ---------------- MOBILE RESPONSIVE FIX ---------------- */
            @media (max-width: 768px) {
                /* Reduce logo size a bit on mobile */
                #osu_logo_img img {
                    max-width: 170px !important;
                    height: auto !important;
                }

                #osu_title_block h2 {
                    font-size: 1.45rem !important;
                    line-height: 1.15 !important;
                    margin-bottom: 6px !important;
                }

                #osu_title_block p {
                    font-size: 0.95rem !important;
                    margin-top: 0px !important;
                    opacity: 0.90;
                }

                /* Dropdown width looks nicer on mobile */
                #topic_dropdown {
                    width: min(340px, 94vw) !important;
                }

                /* Chat height better on phone screens */
                .gr-chatbot { min-height: 55vh !important; }
            }
        </style>
        """)

        # ✅ HEADER ROW 1: Logo + Title only (NO dropdown here)
        with gr.Row():
            gr.Image(
                value="osu_logo.png",
                label="",
                show_label=False,
                interactive=False,
                height=120,
                width=180,
                container=False,
                elem_id="osu_logo_img",
            )

            gr.Markdown(
                "## OSU Virtual TA\nAsk questions about the selected topic.",
                elem_id="osu_title_block",
            )

        # ✅ HEADER ROW 2: Dropdown + Status (separate row fixes mobile squeeze)
        with gr.Row():
            kb_select = gr.Dropdown(
                choices=list(KNOWLEDGE_BASES.keys()),
                value=DEFAULT_TOPIC,
                label="Choose a Topic",
                elem_id="topic_dropdown",
            )

            status = gr.Markdown("Loading topic...")

        gr.Markdown("---")

        # ---- MAIN CHAT ----
        try:
            chat = gr.Chatbot(
                label="Chat",
                height=720,
                latex_delimiters=[{"left": "$$", "right": "$$", "display": True}],
            )
        except TypeError:
            chat = gr.Chatbot(label="Chat", height=720)

        user_box = gr.Textbox(
            placeholder="Ask a question and press Enter...",
            label="Your question",
            lines=2
        )

        with gr.Row():
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear chat")

        # ---- Controls ----
        with gr.Row():
            with gr.Column(scale=1):
                show_sources = gr.Checkbox(True, label="Show sources")
                sources_md = gr.Markdown("", visible=False)

            with gr.Column(scale=1):
                with gr.Accordion("Course / TA info", open=False):
                    gr.Markdown(
                        "- **Course:** ECEN 3314 Electronic Devices & Applications\n"
                        "- **Best practice:** compare answers with real notes.\n"
                        "- **Developer:** Mohamad Ali  OSU Electrical & Computer Engineering\n"
                        "- **Report bugs / request features:** mali14@okstate.edu\n"
                    )

                with gr.Accordion("Practice Quiz Generator", open=False):
                    num_q = gr.Slider(3, 20, value=5, step=1, label="Number of questions")
                    quiz_btn = gr.Button("Generate Quiz")
                    quiz_out = gr.Markdown("")

        problem_image = gr.Image(
            label="Optional problem image (screenshot / photo)",
            type="filepath",
            height=220,
        )

        # ---- CALLBACKS ----
        demo.load(load_topic, inputs=kb_select, outputs=status)
        kb_select.change(load_topic, inputs=kb_select, outputs=status)

        ask_btn.click(
            ask_question,
            inputs=[chat, user_box, show_sources, problem_image],
            outputs=[chat, user_box, sources_md],
        )
        user_box.submit(
            ask_question,
            inputs=[chat, user_box, show_sources, problem_image],
            outputs=[chat, user_box, sources_md],
        )

        clear_btn.click(clear_chat, outputs=[chat, user_box, sources_md])
        quiz_btn.click(generate_quiz, inputs=num_q, outputs=quiz_out)

        gr.Markdown(
            "### Tips\n"
            "- Topic is loaded automatically.\n"
            "- Use 'Show sources' to inspect retrieved text.\n"
            "- Attach a **problem image** above to get vision help in the same chat.\n"
            "- Use 'Practice Quiz' for quick self-tests.\n"
        )

    return demo


# %% [7] Main launcher – robust LAN binding for Gradio 6+



def get_lan_ip():
    """Automatically detect local LAN IP (e.g., 192.168.x.x)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # routing trick; no traffic required
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    demo = build_ui()

    # Allow serving local static asset (optional, but you already use it)
    logo_path = str(Path("osu_logo.png").resolve())
    allowed = [logo_path] if Path(logo_path).exists() else []

    # Hugging Face provides PORT; default to 7860 for local runs
    port = int(os.environ.get("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,                 # HF Spaces already exposes the app
        allowed_paths=allowed,
        theme=OSU_THEME,
        # IMPORTANT: do NOT use prevent_thread_lock=True on Spaces
    )


