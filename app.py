import os
import json
import streamlit as st
import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="VectorTutor • Streamlit", page_icon="📚", layout="wide")
st.title("VectorTutor · Multi‑Agent Study Copilot")
st.caption("Upload materials → Generate flashcards & quizzes → Build a revision plan → Ask doubts")

with st.sidebar:
    st.header("Settings")
    BACKEND_URL = st.text_input("Backend URL", BACKEND_URL)
    st.write("API Keys are read from your environment:")
    st.code("export OPENAI_API_KEY=...\nexport GEMINI_API_KEY=...", language="bash")

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("1) Upload study material")
    upl = st.file_uploader("PDF only", type=["pdf"], accept_multiple_files=False)
    if st.button("Process Material", use_container_width=True, disabled=not upl):
        files = {"file": (upl.name, upl.getvalue(), "application/pdf")}
        data = {"mode": "openai"}
        with st.spinner("Extracting and segmenting topics…"):
            r = requests.post(f"{BACKEND_URL}/api/upload", files=files, data=data, timeout=120)
        if r.ok:
            info = r.json()
            st.session_state["doc_id"] = info.get("document_id")
            st.session_state["topics"] = info.get("topics", [])
            st.success(f"Uploaded: {info.get('title')} · Topics detected: {len(st.session_state['topics'])}")
        else:
            st.error(f"Upload failed: {r.status_code} {r.text}")

    if "topics" in st.session_state and st.session_state["topics"]:
        st.write("Topics:")
        for t in st.session_state["topics"]:
            c1, c2 = st.columns([1,1])
            with c1:
                if st.button(f"Generate Flashcards · {t}", key=f"f-{t}"):
                    payload = {"document_id": st.session_state.get("doc_id"), "topic": t}
                    with st.spinner("Creating flashcards…"):
                        r = requests.post(f"{BACKEND_URL}/api/flashcards", json=payload, timeout=120)
                    if r.ok:
                        cards = r.json().get("flashcards", [])
                        st.success(f"Created {len(cards)} flashcards")
                        with st.expander(f"Preview flashcards for {t}"):
                            for c in cards:
                                st.markdown(f"- Q: {c.get('question')}\n\n  A: {c.get('answer')}")
                    else:
                        st.error(r.text)
            with c2:
                if st.button(f"Generate Quiz · {t}", key=f"q-{t}"):
                    payload = {"document_id": st.session_state.get("doc_id"), "topic": t, "difficulty": "Easy"}
                    with st.spinner("Building quiz…"):
                        r = requests.post(f"{BACKEND_URL}/api/quiz", json=payload, timeout=120)
                    if r.ok:
                        q = r.json()
                        st.session_state["quiz"] = q
                        st.success(f"Quiz ready · {len(q.get('questions', []))} questions")
                    else:
                        st.error(r.text)

with col2:
    st.subheader("Backend status")
    try:
        ping = requests.get(BACKEND_URL, timeout=10).json()
        st.success("Online")
        st.json(ping)
    except Exception as e:
        st.error(f"Offline: {e}")

st.divider()

st.subheader("2) Take quiz and get a plan")
quiz = st.session_state.get("quiz")
answers = []
if quiz and quiz.get("questions"):
    for idx, q in enumerate(quiz["questions"]):
        st.markdown(f"**Q{idx+1}. {q.get('question')}**")
        choice = st.radio("Choose", options=["A","B","C","D"], key=f"ans-{idx}", horizontal=True)
        answers.append({
            "question": q.get("question"),
            "user_answer": choice,
            "answer": q.get("answer"),
            "topic": q.get("topic", "General")
        })
    if st.button("Submit Quiz", type="primary"):
        payload = {"quiz_id": quiz.get("quiz_id"), "answers": answers}
        with st.spinner("Scoring…"):
            r = requests.post(f"{BACKEND_URL}/api/quiz/submit", json=payload, timeout=120)
        if r.ok:
            res = r.json()
            st.success(f"Score: {res.get('score',0):.1f}%")
            wt = res.get('weak_topics', [])
            st.write("Weak topics:", wt)
            if st.button("Build 7‑day plan"):
                with st.spinner("Generating plan…"):
                    pr = requests.post(f"{BACKEND_URL}/api/plan", json={"weak_topics": wt}, timeout=120)
                if pr.ok:
                    plan = pr.json()
                    st.session_state["plan"] = plan
                    st.success("Plan ready")
                else:
                    st.error(pr.text)
        else:
            st.error(r.text)

plan = st.session_state.get("plan")
if plan and plan.get("schedule"):
    st.subheader("Your adaptive plan")
    for day in plan["schedule"]:
        with st.expander(day.get("day", "Day")):
            st.write("Topics:", day.get("topics", []))
            st.write("Tasks:", day.get("tasks", []))
            st.write("Duration (min):", day.get("duration_minutes", 30))

st.divider()

st.subheader("3) Ask a doubt")
q = st.text_input("Question about your material")
if st.button("Ask"):
    payload = {"document_id": st.session_state.get("doc_id", ""), "question": q}
    with st.spinner("Thinking…"):
        r = requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=120)
    if r.ok:
        st.info(r.json().get("answer"))
    else:
        st.error(r.text)
