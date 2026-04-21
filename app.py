import streamlit as st
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------
# 🔹 Fetch Repo Files
# -------------------------------
def get_repo_files(repo):
    url = f"https://api.github.com/repos/{repo}/contents"
    res = requests.get(url)

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
# 🔹 Build FAISS DB
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
# 🔹 Generate Questions (Offline)
# -------------------------------
def generate_questions():
    return [
        "Explain the main functionality of your project.",
        "Describe the core logic used in your code.",
        "What technologies or libraries are used and why?",
        "What are the limitations of your implementation?",
        "How can this project be improved?"
    ]

# -------------------------------
# 🔹 Evaluate Answer (Simple Scoring)
# -------------------------------
def evaluate_answer(context, answer):
    score = 0

    # Basic keyword matching
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())

    match_ratio = len(context_words & answer_words) / (len(context_words) + 1)

    if match_ratio > 0.2:
        score = 3
    elif match_ratio > 0.1:
        score = 2
    else:
        score = 1

    return score

# -------------------------------
# 🔹 UI
# -------------------------------
st.title("🎤 Offline RAG Viva (No API, No Limits)")

repo = st.text_input("Enter GitHub Repo (username/repo)")

if st.button("Start Viva"):
    code = get_repo_files(repo)

    if not code:
        st.error("❌ Could not fetch repo")
    else:
        code = code[:5000]
        db = build_vector_db(code)

        st.session_state["db"] = db
        st.session_state["questions"] = generate_questions()

# -------------------------------
# 🔹 Questions
# -------------------------------
if "questions" in st.session_state:
    st.markdown("## 📝 Answer Questions")

    answers = []

    for i, q in enumerate(st.session_state["questions"]):
        st.write(f"Q{i+1}: {q}")
        ans = st.text_input(f"Answer {i+1}", key=f"ans_{i}")
        answers.append(ans)

    if st.button("Submit Viva"):
        total = 0

        st.markdown("## 📊 Evaluation")

        for i, q in enumerate(st.session_state["questions"]):
            context = st.session_state["db"].similarity_search(q, k=1)[0].page_content

            score = evaluate_answer(context, answers[i])

            st.write(f"Q{i+1}: Score {score}/3")
            total += score

        st.markdown("---")
        st.subheader(f"🎯 Viva Marks: {total} / 15")
