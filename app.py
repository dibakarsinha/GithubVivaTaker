import streamlit as st
import requests
import os
import time

from openai import OpenAI, RateLimitError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------
# 🔑 Optional OpenAI (Hybrid)
# -------------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

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
# 🔹 Build FAISS
# -------------------------------
@st.cache_resource
def build_vector_db(code_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(code_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)

# -------------------------------
# 🔹 Questions (Offline)
# -------------------------------
def generate_questions():
    return [
        "Explain the main functionality of your project.",
        "Describe the core logic used.",
        "Why did you choose this approach?",
        "What are limitations?",
        "How can this project be improved?"
    ]

# -------------------------------
# 🔹 Offline Evaluation
# -------------------------------
def evaluate_offline(context, answer):
    context_words = set(context.lower().split())
    answer_words = set(answer.lower().split())

    match_ratio = len(context_words & answer_words) / (len(context_words) + 1)

    if match_ratio > 0.2:
        return 3
    elif match_ratio > 0.1:
        return 2
    else:
        return 1

# -------------------------------
# 🔹 AI Evaluation (Optional)
# -------------------------------
def evaluate_ai(context, question, answer):
    if not client:
        return "AI not enabled"

    prompt = f"""
    Context:
    {context[:1500]}

    Question: {question}
    Answer: {answer}

    Score out of 3 and justify.
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content

    except RateLimitError:
        return "⚠️ Rate limit reached"

# -------------------------------
# 🔹 UI
# -------------------------------
st.title("🎤 Hybrid Viva (Offline + AI)")

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
        st.session_state["questions"] = generate_questions()

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
    # Submit Offline
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

        st.session_state["offline_total"] = total
        st.session_state["answers"] = answers

        st.markdown("---")
        st.subheader(f"🎯 Offline Marks: {total} / 15")

    # -------------------------------
    # AI Upgrade Button
    # -------------------------------
    if "offline_total" in st.session_state:
        st.markdown("## 🤖 Optional AI Evaluation")

        if st.button("Upgrade Evaluation (AI)"):
            st.markdown("### AI Feedback")

            for i, q in enumerate(st.session_state["questions"]):
                feedback = evaluate_ai(
                    st.session_state["contexts"][i],
                    q,
                    st.session_state["answers"][i]
                )
                st.write(f"Q{i+1}: {feedback}")
