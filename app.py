import streamlit as st
import requests
import os

from openai import OpenAI

# RAG imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# 🔑 ENV VARIABLES
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        if file["type"] == "file" and file["name"].endswith((".py", ".js", ".html", ".ipynb")):
            file_res = requests.get(file["download_url"])
            if file_res.status_code == 200:
                contents.append(file_res.text[:2000])  # limit size

    return "\n".join(contents)

# -------------------------------
# 🔹 Build FAISS Vector DB
# -------------------------------
@st.cache_resource
def build_vector_db(code_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(code_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)

    return db

# -------------------------------
# 🔹 Retrieve Relevant Context
# -------------------------------
def get_context(db, query):
    docs = db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# -------------------------------
# 🔹 Generate Viva Questions
# -------------------------------
def generate_questions(db):
    context = get_context(db, "overall project functionality")

    prompt = f"""
    Based on this project code, generate 5 viva questions:
    - 2 easy
    - 2 medium
    - 1 hard
    Focus on logic and implementation only.

    Context:
    {context}
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    questions = res.choices[0].message.content.split("\n")

    # clean empty lines
    return [q.strip() for q in questions if q.strip()]

# -------------------------------
# 🔹 Evaluate Student Answer
# -------------------------------
def evaluate_answer(db, question, answer):
    context = get_context(db, question)

    prompt = f"""
    You are evaluating a student's viva answer.

    Context from code:
    {context}

    Question: {question}
    Student Answer: {answer}

    Evaluate strictly based on context.

    Give:
    Score: X/3
    Reason: short explanation
    """

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content

# -------------------------------
# 🔹 STREAMLIT UI
# -------------------------------
st.title("🎤 RAG-Based Viva Evaluation System")

repo = st.text_input("Enter GitHub Repo (username/repo)")

# -------- Start Viva --------
if st.button("Generate Viva Questions"):
    code = get_repo_files(repo)

    if not code:
        st.error("❌ Could not fetch repo or repo is private")
    else:
        db = build_vector_db(code)
        st.session_state["db"] = db
        st.session_state["questions"] = generate_questions(db)

        st.success("✅ Questions Generated")

# -------- Display Questions --------
if "questions" in st.session_state:
    st.markdown("## 📝 Answer the Following Questions")

    answers = []

    for i, q in enumerate(st.session_state["questions"]):
        st.write(f"Q{i+1}: {q}")
        ans = st.text_input(f"Answer {i+1}", key=f"ans_{i}")
        answers.append(ans)

    # -------- Submit --------
    if st.button("Submit Viva"):
        total_score = 0

        st.markdown("## 📊 Evaluation")

        for i, q in enumerate(st.session_state["questions"]):
            result = evaluate_answer(
                st.session_state["db"],
                q,
                answers[i]
            )

            st.write(f"**Q{i+1}:** {result}")

            # simple score extraction
            if "3" in result:
                total_score += 3
            elif "2" in result:
                total_score += 2
            else:
                total_score += 1

        st.markdown("---")
        st.subheader(f"🎯 Viva Marks: {total_score} / 15")
