import streamlit as st
import requests
import time

# Gemini
import google.generativeai as genai

# RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------
# 🔑 Gemini Setup
# -------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
else:
    model = None

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
# 🔹 Build Vector DB (FAISS)
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
    docs = db.similarity_search(query, k=2)
    return "\n".join([d.page_content for d in docs])

# -------------------------------
# 🔹 Offline Questions
# -------------------------------
def offline_questions():
    return [
        "Explain the main functionality of your project.",
        "Describe the core logic used.",
        "What technologies or libraries are used and why?",
        "What are the limitations of your implementation?",
        "How can this project be improved?"
    ]

# -------------------------------
# 🔹 Gemini Question Generation
# -------------------------------
def generate_questions_ai(context):
    if not model:
        return None

    prompt = f"""
    Generate exactly 5 viva questions:
    - 2 easy
    - 2 medium
    - 1 hard

    Based on this project:
    {context[:2000]}
    """

    try:
        response = model.generate_content(prompt)
        qs = [q.strip() for q in response.text.split("\n") if q.strip()]
        return qs if len(qs) >= 5 else None
    except:
        return None

# -------------------------------
# 🔹 Offline Evaluation
# -------------------------------
def evaluate_offline(context, answer):
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())

    overlap = len(context_words & answer_words)
    ratio = overlap / (len(context_words) + 1)

    if ratio > 0.2:
        return 3
    elif ratio > 0.1:
        return 2
    else:
        return 1

# -------------------------------
# 🔹 Gemini Evaluation
# -------------------------------
def evaluate_ai(context, question, answer):
    if not model:
        return "AI not enabled"

    prompt = f"""
    Evaluate the student's answer.

    Context:
    {context[:1500]}

    Question: {question}
    Answer: {answer}

    Output:
    Score: X/3
    Reason: short explanation
    """

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"

# -------------------------------
# 🔹 UI
# -------------------------------
st.title("🎤 Hybrid Viva (Offline + Gemini AI)")

repo = st.text_input("Enter GitHub Repo (username/repo)")

# -------------------------------
# Start Viva
# -------------------------------
if st.button("Start Viva"):
    code = get_repo_files(repo)

    if not code:
        st.error("❌ Could not fetch repo")
    else:
        code = code[:5000]

        db = build_vector_db(code)
        st.session_state["db"] = db

        context = get_context(db, "project functionality")

        # Try AI questions first
        questions = generate_questions_ai(context)

        # Fallback
        if not questions:
            questions = offline_questions()

        st.session_state["questions"] = questions
        st.success("✅ Questions Ready")

# -------------------------------
# Show Questions
# -------------------------------
if "questions" in st.session_state:
    st.markdown("## 📝 Answer Questions")

    answers = []

    for i, q in enumerate(st.session_state["questions"]):
        st.write(f"Q{i+1}: {q}")
        ans = st.text_input(f"Answer {i+1}", key=f"ans_{i}")
        answers.append(ans)

    # -------------------------------
    # Offline Evaluation
    # -------------------------------
    if st.button("Submit (Offline Evaluation)"):
        total = 0
        st.session_state["contexts"] = []

        st.markdown("## 📊 Offline Evaluation")

        for i, q in enumerate(st.session_state["questions"]):
            context = st.session_state["db"].similarity_search(q, k=1)[0].page_content
            st.session_state["contexts"].append(context)

            score = evaluate_offline(context, answers[i])
            st.write(f"Q{i+1}: Score {score}/3")

            total += score

        st.session_state["answers"] = answers
        st.session_state["offline_total"] = total

        st.markdown("---")
        st.subheader(f"🎯 Offline Marks: {total} / 15")

    # -------------------------------
    # AI Upgrade
    # -------------------------------
    if "offline_total" in st.session_state:
        st.markdown("## 🤖 AI Evaluation (Gemini)")

        if st.button("Upgrade Evaluation"):
            for i, q in enumerate(st.session_state["questions"]):
                feedback = evaluate_ai(
                    st.session_state["contexts"][i],
                    q,
                    st.session_state["answers"][i]
                )
                st.write(f"Q{i+1}: {feedback}")
