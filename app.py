import streamlit as st
import requests
import os
import time

from openai import OpenAI, RateLimitError

# RAG imports (updated)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------
# 🔑 Secrets (Streamlit Cloud)
# -------------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# 🔹 Fetch Repo Files
# -------------------------------
def get_repo_files(repo):
    url = f"https://api.github.com/repos/{repo}/contents"
    res = requests.get(url, headers=HEADERS)

    if res.status_code != 200:
        return ""

    contents = []
    for file in res.json():
        if file["type"] == "file" and file["name"].endswith((".py", ".js", ".html")):
            file_res = requests.get(file["download_url"])
            if file_res.status_code == 200:
                contents.append(file_res.text[:2000])

    return "\n".join(contents)

# -------------------------------
# 🔹 Build FAISS DB (LOCAL EMBEDDINGS)
# -------------------------------
@st.cache_resource
def build_vector_db(code_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(code_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)
    return db

# -------------------------------
# 🔹 Retrieve Context
# -------------------------------
def get_context(db, query):
    docs = db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# -------------------------------
# 🔹 Fallback Questions
# -------------------------------
def fallback_questions():
    return [
        "Explain the main functionality of your project.",
        "What is the core logic used?",
        "Why did you choose this approach?",
        "What are limitations?",
        "How can this be improved?"
    ]

# -------------------------------
# 🔹 Generate Questions (with retry)
# -------------------------------
def generate_questions_llm(context):
    prompt = f"""
    Generate 5 viva questions (2 easy, 2 medium, 1 hard)
    based on this code:

    {context[:2000]}
    """

    for attempt in range(5):
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            return [q.strip() for q in res.choices[0].message.content.split("\n") if q.strip()]

        except RateLimitError:
            time.sleep(2 ** attempt)

    return fallback_questions()

# -------------------------------
# 🔹 Cache Questions
# -------------------------------
@st.cache_data(show_spinner=False)
def generate_questions_cached(context):
    return generate_questions_llm(context)

# -------------------------------
# 🔹 Evaluate Answer
# -------------------------------
def evaluate_answer(db, question, answer):
    context = get_context(db, question)

    prompt = f"""
    Context:
    {context[:1500]}

    Question: {question}
    Answer: {answer}

    Score out of 3.
    Format:
    Score: X
    Reason: short
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    except RateLimitError:
        return "Score: 1\nReason: Could not evaluate due to rate limit"

# -------------------------------
# 🔹 UI
# -------------------------------
st.title("🎤 RAG Viva Evaluation (Stable Version)")

repo = st.text_input("Enter GitHub Repo (username/repo)")

# Throttle control
if "last_call" not in st.session_state:
    st.session_state["last_call"] = 0

# -------------------------------
# 🔹 Generate Questions
# -------------------------------
if st.button("Generate Viva Questions"):
    now = time.time()

    if now - st.session_state["last_call"] < 5:
        st.warning("Please wait before generating again")
    else:
        st.session_state["last_call"] = now

        code = get_repo_files(repo)

        if not code:
            st.error("❌ Could not fetch repo")
        else:
            code = code[:5000]

            db = build_vector_db(code)
            st.session_state["db"] = db

            context = get_context(db, "project functionality")

            questions = generate_questions_cached(context)

            st.session_state["questions"] = questions
            st.success("✅ Questions Ready")

# -------------------------------
# 🔹 Display Questions
# -------------------------------
if "questions" in st.session_state:
    st.markdown("## 📝 Answer Questions")

    answers = []

    for i, q in enumerate(st.session_state["questions"]):
        st.write(f"Q{i+1}: {q}")
        ans = st.text_input(f"Answer {i+1}", key=f"ans_{i}")
        answers.append(ans)

    # -------------------------------
    # 🔹 Submit
    # -------------------------------
    if st.button("Submit Viva"):
        total = 0

        st.markdown("## 📊 Evaluation")

        for i, q in enumerate(st.session_state["questions"]):
            result = evaluate_answer(st.session_state["db"], q, answers[i])

            st.write(f"**Q{i+1}:** {result}")

            if "3" in result:
                total += 3
            elif "2" in result:
                total += 2
            else:
                total += 1

        st.markdown("---")
        st.subheader(f"🎯 Viva Marks: {total} / 15")
