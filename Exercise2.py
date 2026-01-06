
import os
import re
import time
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_text_from_pdf(file) -> str:
    reader = PdfReader(file)
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return clean_text("\n".join(parts))

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))

load_dotenv()

st.set_page_config(page_title="Exercise 2 - Building your own Chatbot ", layout="wide")
st.title("Exercise 2: Building your own Chatbot (App for summarizing documents)")

st.markdown("""
This chatbot can be used in two ways:

• Without uploading a PDF: the chatbot works as a general conversational assistant using a large language model and conversational memory.

• With an uploaded PDF: the chatbot activates Retrieval-Augmented Generation (RAG), allowing it to answer questions, generate summaries, and provide critical opinions based exclusively on the content of the document.

────────────────────────────────────────

How to use the left sidebar (controls panel):

1) Parameters & Evaluation
- Temperature: controls the creativity of the responses. Lower values produce more precise and factual answers, while higher values generate more expressive and abstract responses.
- Top-k chunks: defines how many text segments are retrieved from the document to answer a question. Higher values increase context coverage but may introduce noise.
- Chunk size: determines the length of text segments used to index the document. Larger chunks preserve context; smaller chunks increase retrieval precision.
- Chunk overlap: defines how much consecutive chunks overlap, helping preserve continuity between text segments.

2) Checkboxes
- Show retrieved context (debug): displays the document fragments retrieved by the system to generate the response. This is useful for understanding and evaluating the RAG process.
- Faithfulness evaluation: activates an automatic evaluation that checks whether the generated answer is supported by the retrieved document context.

3) External Knowledge Base (PDF)
- Use the file uploader to upload a PDF document (e.g., articles, books, reports).
- Once uploaded, the document is automatically processed, split into chunks, embedded, and stored in a vector database.

4) Mode selection (dropdown list)
- General Chat (no document): standard conversational chatbot without external knowledge.
- Ask the Document (RAG): answers questions strictly based on the uploaded document.
- Summarize Document: generates a summary of the document using retrieved content.
- Opinion / Critique: provides a critical analysis of the document, including strengths, weaknesses, and recommendations, grounded in the document text.

The chatbot maintains conversational memory across interactions and displays performance metrics such as response latency and retrieval usage.
""")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is missing. Configure it in secrets as in exercise 1.")
    st.stop()

st.sidebar.header("Tuning & Evaluation")

temperature = st.sidebar.slider("Temperature (creatividad)", 0.0, 1.0, 0.3, 0.1)
k = st.sidebar.slider("Top-k chunks recuperados", 1, 10, 4, 1)
chunk_size = st.sidebar.slider("Chunk size", 300, 1500, 900, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 120, 20)

show_context = st.sidebar.checkbox("Mostrar contexto recuperado (debug)", value=False)
faithfulness_check = st.sidebar.checkbox("Evaluar 'faithfulness' (LLM-as-judge)", value=False)

st.sidebar.divider()

st.sidebar.subheader("External Knowledge Base (PDF)")
uploaded_pdf = st.sidebar.file_uploader("Sube un PDF (opcional)", type=["pdf"])

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

if "ui_history" not in st.session_state:
    st.session_state.ui_history = []

if "doc_text" not in st.session_state:
    st.session_state.doc_text = ""
if "doc_words" not in st.session_state:
    st.session_state.doc_words = 0
if "db" not in st.session_state:
    st.session_state.db = None
if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

@st.cache_resource(show_spinner=False)
def build_vectorstore_from_pdf_bytes(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int):
    
    from io import BytesIO
    pdf_file = BytesIO(pdf_bytes)
    text = extract_text_from_pdf(pdf_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)

    db = Chroma.from_texts(texts=chunks, embedding=embeddings)
    return db, text, len(chunks)

if uploaded_pdf is not None:
    pdf_bytes = uploaded_pdf.read()
    db, doc_text, n_chunks = build_vectorstore_from_pdf_bytes(pdf_bytes, chunk_size, chunk_overlap)
    st.session_state.db = db
    st.session_state.doc_text = doc_text
    st.session_state.doc_words = word_count(doc_text)
    st.session_state.chunks_count = n_chunks
    st.sidebar.success(f"PDF indexado: {n_chunks} chunks | ~{st.session_state.doc_words} palabras")

mode = st.sidebar.selectbox(
    "Modo",
    [
        "General Chat (sin documento)",
        "Ask the Document (RAG)",
        "Summarize Document",
        "Opinion / Critique (based on document)"
    ],
)

def retrieve_context(query: str, top_k: int):
    if st.session_state.db is None:
        return "", []

    retriever = st.session_state.db.as_retriever(search_kwargs={"k": top_k})

    # LangChain moderno: invoke()
    try:
        docs = retriever.invoke(query)
    except Exception:
        docs = retriever.get_relevant_documents(query)

    context = "\n".join([getattr(d, "page_content", str(d)) for d in docs])
    return context, docs

def build_prompt_general(user_input: str):
    history = st.session_state.memory.buffer
    prompt = f"""
You are a helpful assistant.

Conversation history:
{history}

User message:
{user_input}

Answer:
""".strip()
    return prompt

def build_prompt_rag(user_input: str, context: str):
    history = st.session_state.memory.buffer
    prompt = f"""
You are a helpful assistant.
Use ONLY the retrieved context to answer. If the answer is not in the context, say:
"I don't know based on the provided document, please tell me a question about the document."

Conversation history:
{history}

Retrieved context:
{context}

User question:
{user_input}

Answer:
""".strip()
    return prompt

def build_prompt_summary(style: str, context: str, max_words: int | None):
    constraint = f"Limit the summary to at most {max_words} words." if max_words else ""
    prompt = f"""
You are a helpful assistant.
Summarize the document using ONLY the retrieved context.

Summary style:
{style}

{constraint}

Retrieved context:
{context}

Return the summary now:
""".strip()
    return prompt

def build_prompt_opinion(context: str):
    prompt = f"""
You are a critical reviewer.
Provide an opinion/critique of the document using ONLY the retrieved context.
Do NOT invent facts. If something is not supported by the context, explicitly say so.

Your output MUST have these sections:
1) Document purpose (inferred from context)
2) Strengths (with brief evidence)
3) Weaknesses / gaps (with brief evidence)
4) Recommendations (specific improvements)
5) Open questions

Use short evidence snippets (max 1 sentence each), not long quotes.

Retrieved context:
{context}

Critique:
""".strip()
    return prompt

def judge_faithfulness(answer: str, context: str) -> str:
    
    judge_prompt = f"""
You are an evaluator. Rate how well the ANSWER is supported by the CONTEXT.
Scale:
1 = unsupported / hallucinated
3 = partially supported
5 = fully supported

Return:
- score: (1-5)
- brief justification (1-2 lines)

CONTEXT:
{context}

ANSWER:
{answer}
""".strip()
    return llm.invoke(judge_prompt).content

summary_style = None
summary_max_words = None

if mode == "Summarize Document":
    col1, col2 = st.columns(2)
    with col1:
        summary_style = st.selectbox(
            "Estilo de resumen",
            ["Concise", "Executive", "Bullet points", "Focused on conclusions"]
        )
    with col2:
        summary_max_words = st.number_input("Máx. palabras (opcional)", min_value=0, value=180, step=20)
        if summary_max_words == 0:
            summary_max_words = None

user_input = st.chat_input("Write your message...")

def respond(user_input: str):
    t0 = time.time()

    used_rag = 0
    context = ""
    answer = ""
    judge = None
    
    if mode == "General Chat (sin documento)" or st.session_state.db is None:
        prompt = build_prompt_general(user_input)
        answer = llm.invoke(prompt).content
        st.session_state.memory.save_context({"input": user_input}, {"output": answer})

    elif mode == "Ask the Document (RAG)":
        context, _docs = retrieve_context(user_input, k)
        used_rag = 1 if context.strip() else 0
        prompt = build_prompt_rag(user_input, context)
        answer = llm.invoke(prompt).content
        st.session_state.memory.save_context({"input": user_input}, {"output": answer})

    elif mode == "Summarize Document":
        summary_query = f"main ideas, key points, conclusions, purpose. user request: {user_input}"
        context, _docs = retrieve_context(summary_query, k)
        used_rag = 1 if context.strip() else 0

        prompt = build_prompt_summary(summary_style or "Concise", context, summary_max_words)
        answer = llm.invoke(prompt).content

        st.session_state.memory.save_context({"input": f"[SUMMARY REQUEST] {user_input}"}, {"output": answer})

    elif mode == "Opinion / Critique (based on document)":
        critique_query = "purpose, thesis, arguments, evidence, limitations, methodology, conclusions"
        context, _docs = retrieve_context(critique_query, k)
        used_rag = 1 if context.strip() else 0

        prompt = build_prompt_opinion(context)
        answer = llm.invoke(prompt).content

        st.session_state.memory.save_context({"input": "[OPINION REQUEST]"}, {"output": "Opinion generated."})

    latency = time.time() - t0

    metrics = {
        "latency_s": round(latency, 2),
        "used_rag": used_rag,
        "k": k,
        "temperature": temperature
    }
   
    if mode == "Summarize Document" and st.session_state.doc_words > 0:
        summary_words = word_count(answer)
        metrics["compression_ratio"] = round(summary_words / st.session_state.doc_words, 4)
    
    if faithfulness_check and context.strip():
        judge = judge_faithfulness(answer, context)

    return answer, metrics, context, judge

for role, msg in st.session_state.ui_history:
    with st.chat_message(role):
        st.write(msg)

if user_input:
    st.session_state.ui_history.append(("user", user_input))
    with st.chat_message("user"):
        st.write(user_input)
    
    answer, metrics, ctx, judge = respond(user_input)

    with st.chat_message("assistant"):
        if st.session_state.db is None and mode != "General Chat (sin documento)":
            st.info("No PDF uploaded yet; answering using general knowledge (no RAG).")

        st.write(answer)
        st.caption(
            f"Latency: {metrics.get('latency_s')}s | used_rag={metrics.get('used_rag')} | "
            f"k={metrics.get('k')} | temp={metrics.get('temperature')}"
            + (f" | compression_ratio={metrics.get('compression_ratio')}" if "compression_ratio" in metrics else "")
        )

        if judge:
            st.subheader("Faithfulness (auto-evaluation)")
            st.write(judge)

    st.session_state.ui_history.append(("assistant", answer))
    
    if show_context and ctx.strip():
        st.sidebar.subheader("Retrieved context (debug)")
        st.sidebar.write(ctx)
