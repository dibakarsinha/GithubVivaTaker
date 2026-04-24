import streamlit as st
import requests
from datetime import datetime
import pandas as pd

# Gemini
import google.generativeai as genai

# RAG
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# -------------------------------
# 🔑 CONFIG
# -------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", None)
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        st.write(m.name)
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-pro")
else:
    model = None

# -------------------------------
# 🔹 GOOGLE SHEETS
# -------------------------------
def connect_sheet():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ],
    )
    client = gspread.authorize(creds)
    SHEET_ID = "1V1-NcPe2zsKa-eTXnWhTOewJVsCAoeM5WlYqKjREcPM"
    return client.open_by_key(SHEET_ID).sheet1
    

def save_to_sheet(name, repo, marks):
    sheet = connect_sheet()
    sheet.append_row([
        name,
        repo,
        marks,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ])

def load_sheet():
    sheet = connect_sheet()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# -------------------------------
# 🔹 GITHUB FETCH
# -------------------------------
def get_repo_files(repo):
    headers = {}

    # Optional: GitHub token
    if "GITHUB_TOKEN" in st.secrets:
        headers["Authorization"] = f"Bearer {st.secrets['GITHUB_TOKEN']}"

    def fetch(url):
        res = requests.get(url, headers=headers)

        st.write("Fetching:", url)
        st.write("Status:", res.status_code)

        if res.status_code != 200:
            st.error(f"GitHub API Error: {res.text}")
            return []

        data = res.json()
        all_files = []

        for item in data:
            if item["type"] == "file":
                if item["name"].endswith((".py", ".js", ".html", ".java", ".cpp")):
                    file_res = requests.get(item["download_url"])
                    if file_res.status_code == 200:
                        all_files.append(file_res.text[:2000])

            elif item["type"] == "dir":
                all_files.extend(fetch(item["url"]))  # 🔥 recursion

        return all_files

    url = f"https://api.github.com/repos/{repo}/contents"
    files = fetch(url)

    st.write("Total files fetched:", len(files))

    return "\n".join(files)
# -------------------------------
# 🔹 VECTOR DB
# -------------------------------
@st.cache_resource
def build_db(code):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(code)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(chunks, embeddings)

def get_context(db, query):
    docs = db.similarity_search(query, k=2)
    return "\n".join([d.page_content for d in docs])

# -------------------------------
# 🔹 QUESTIONS
# -------------------------------
def offline_questions():
    return [
        "Explain your project.",
        "What is core logic?",
        "Why this approach?",
        "Limitations?",
        "Improvements?"
    ]

def generate_questions_ai(context):
    if not model:
        return None

    try:
        res = model.generate_content(f"Generate 5 viva questions:\n{context[:2000]}")
        qs = [q.strip() for q in res.text.split("\n") if q.strip()]
        return qs if len(qs) >= 5 else None
    except:
        return None

# -------------------------------
# 🔹 EVALUATION
# -------------------------------
def eval_offline(context, ans):
    cw = set(context.lower().split())
    aw = set(ans.lower().split())
    ratio = len(cw & aw) / (len(cw) + 1)

    if ratio > 0.2:
        return 3
    elif ratio > 0.1:
        return 2
    elif ratio > 0.05:
        return 1
    else:
        return 0

def eval_ai(context, q, a):
    if not model:
        return "❌ AI disabled (check API key)"

    if not context.strip():
        return "❌ Empty context (repo issue)"

    try:
        prompt = f"""
        Context:
        {context[:1000]}

        Question:
        {q}

        Answer:
        {a}

        Give score out of 3 with reason.
        """

        res = model.generate_content(prompt)
        return res.text

    except Exception as e:
        return f"❌ AI error: {str(e)}"

# -------------------------------
# 🔹 UI NAVIGATION
# -------------------------------
menu = st.sidebar.radio("Navigation", ["Student Viva", "Faculty Dashboard"])

# ===============================
# 🎓 STUDENT SIDE
# ===============================
if menu == "Student Viva":

    st.title("🎤 Viva Evaluation System")

    name = st.text_input("Student Name")
    repo = st.text_input("GitHub Repo (username/repo)")

    if st.button("Start Viva"):
        code = get_repo_files(repo)

        if not code:
            st.error("❌ Repo fetch failed")
        else:
            code = code[:5000]
            db = build_db(code)

            st.session_state["db"] = db
            context = get_context(db, "project")

            qs = generate_questions_ai(context)
            if not qs:
                qs = offline_questions()

            st.session_state["qs"] = qs

    if "qs" in st.session_state:
        answers = []

        st.markdown("## Questions")

        for i, q in enumerate(st.session_state["qs"]):
            st.write(f"Q{i+1}: {q}")
            ans = st.text_input(f"Answer {i+1}", key=f"a{i}")
            answers.append(ans)

        if st.button("Submit Viva"):
            total = 0
            contexts = []

            for i, q in enumerate(st.session_state["qs"]):
                ctx = st.session_state["db"].similarity_search(q, k=1)[0].page_content
                contexts.append(ctx)

                score = eval_offline(ctx, answers[i])
                total += score

                st.write(f"Q{i+1}: {score}/3")

            st.subheader(f"Total: {total}/15")

            save_to_sheet(name, repo, total)
            st.success("✅ Saved to Google Sheets")

            st.session_state["contexts"] = contexts
            st.session_state["answers"] = answers
            st.session_state["total"] = total

    if "total" in st.session_state:
        if st.button("Upgrade with AI"):
            st.markdown("### AI Evaluation")

            for i, q in enumerate(st.session_state["qs"]):
                fb = eval_ai(
                    st.session_state["contexts"][i],
                    q,
                    st.session_state["answers"][i]
                )
                st.write(f"Q{i+1}: {fb}")

# ===============================
# 📊 FACULTY DASHBOARD
# ===============================
if menu == "Faculty Dashboard":

    st.title("📊 Faculty Dashboard")

    if st.button("Load Data"):
        df = load_sheet()

        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "viva_marks.csv"
        )
