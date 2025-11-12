import os
import io
import json
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import db, create_document, get_documents
from schemas import Document, Flashcard, Quiz, QuizResult, Plan

# LLM providers (optional imports with graceful fallback)
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

# PDF utils (optional)
try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

app = FastAPI(title="VectorTutor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_MODE = os.getenv("LLM_MODE", "openai")  # openai | gemini
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY and genai is not None:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass

openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None


class UploadResponse(BaseModel):
    document_id: str
    title: str
    topics: List[str]


# ----------------- Utility functions -----------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        # Fallback: no PDF parser available; allow flow to continue
        return ""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(texts)
    except Exception as e:
        # Non-fatal; continue with empty text which will still let LLM produce structure
        return ""


def call_llm(prompt: str, temperature: float = 0.2) -> str:
    mode = BACKEND_MODE
    # Gemini path
    if mode == "gemini" and GEMINI_API_KEY and genai is not None:
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            resp = model.generate_content(prompt)
            return getattr(resp, "text", None) or ""
        except Exception:
            pass
    # OpenAI path
    if openai_client is not None:
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception:
            pass
    # Fallback mock
    return "Summary/QA placeholder based on prompt: " + prompt[:200]


# ----------------- Endpoints -----------------

@app.get("/")
def root():
    return {"message": "VectorTutor Backend Running"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_material(
    file: UploadFile = File(...),
    mode: Optional[str] = Form(default="openai")
):
    data = await file.read()
    text = extract_text_from_pdf(data)

    # topic segmentation via LLM
    prompt = (
        "You are a study material topic segmenter. Split the following text into 6-12 high-level topics. "
        "Return JSON with keys: topics (list of strings). Text:\n\n" + text[:6000]
    )
    try:
        topics_raw = call_llm(prompt)
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(topics_raw)
            topics = parsed.get("topics", [])
        except Exception:
            # heuristic extraction
            topics = [t.strip("-• ") for t in topics_raw.split("\n") if 2 <= len(t) <= 80][:10]
    except Exception:
        topics = []

    doc = Document(title=file.filename, source_filename=file.filename, mode=mode, topics=topics)
    doc_id = create_document("document", doc)

    return UploadResponse(document_id=doc_id, title=file.filename, topics=topics)


class FlashReq(BaseModel):
    document_id: str
    topic: Optional[str] = None


@app.post("/api/flashcards")
async def generate_flashcards(req: FlashReq):
    # fetch document text (not stored fully; in production store chunks)
    docs = get_documents("document", {"_id": {"$exists": True}})
    title = next((d.get("title") for d in docs if str(d.get("_id")) == req.document_id), "Document")

    prompt = f"Create 12 concise Q/A flashcards in JSON list for topic '{req.topic or 'All'}' from the document titled {title}. Keys: question, answer."
    output = call_llm(prompt)

    try:
        cards = json.loads(output)
        if isinstance(cards, dict) and "flashcards" in cards:
            cards = cards["flashcards"]
    except Exception:
        # fallback simple parsing
        cards = [{"question": q.strip(), "answer": ""} for q in output.split("\n") if q.strip()][:12]

    # store
    for c in cards:
        f = Flashcard(document_id=req.document_id, topic=req.topic, question=c.get("question", ""), answer=c.get("answer", ""))
        create_document("flashcard", f)

    return {"flashcards": cards[:20]}


class QuizReq(BaseModel):
    document_id: str
    topic: Optional[str] = None
    difficulty: str = "Easy"


@app.post("/api/quiz")
async def generate_quiz(req: QuizReq):
    prompt = (
        f"Create a {req.difficulty} 10-question multiple-choice quiz as JSON with list under 'questions'. "
        f"Each question has fields: question, options (A-D), answer, explanation. Topic: {req.topic or 'All'}"
    )
    output = call_llm(prompt)
    try:
        parsed = json.loads(output)
        questions = parsed.get("questions", [])
    except Exception:
        questions = []

    qz = Quiz(document_id=req.document_id, topic=req.topic, difficulty=req.difficulty, questions=questions)
    qid = create_document("quiz", qz)

    return {"quiz_id": qid, "questions": questions}


class QuizSubmit(BaseModel):
    quiz_id: str
    answers: List[Dict[str, Any]]


@app.post("/api/quiz/submit")
async def submit_quiz(req: QuizSubmit):
    # simple scoring
    # In production, fetch quiz by id and compare
    correct = 0
    total = len(req.answers)
    weak_topics = []

    for a in req.answers:
        if str(a.get("user_answer")).strip().upper() == str(a.get("answer", "")).strip().upper():
            correct += 1
        else:
            wt = a.get("topic") or "General"
            weak_topics.append(wt)

    score = (correct / total) * 100 if total else 0
    res = QuizResult(quiz_id=req.quiz_id, score=score, answers=req.answers, weak_topics=weak_topics)
    rid = create_document("quizresult", res)

    return {"score": score, "weak_topics": weak_topics, "result_id": rid}


class PlanReq(BaseModel):
    user_id: Optional[str] = None
    weak_topics: List[str] = []


@app.post("/api/plan")
async def build_plan(req: PlanReq):
    prompt = (
        "Build a 7-day adaptive revision plan as JSON list under 'schedule'. Each item: day, topics (list), tasks (list), duration_minutes (int). "
        f"Focus on weak topics: {', '.join(req.weak_topics) if req.weak_topics else 'General review'}."
    )
    output = call_llm(prompt)
    try:
        parsed = json.loads(output)
        schedule = parsed.get("schedule", [])
    except Exception:
        schedule = []

    plan = Plan(user_id=req.user_id, schedule=schedule)
    pid = create_document("plan", plan)

    return {"plan_id": pid, "schedule": schedule}


class ChatReq(BaseModel):
    document_id: str
    question: str


@app.post("/api/chat")
async def chat(req: ChatReq):
    prompt = (
        "Answer the question using only the uploaded material context. If unsure, say you don't have enough context. "
        f"Question: {req.question}\nProvide a concise answer with a brief reference to likely sections."
    )
    answer = call_llm(prompt)
    return {"answer": answer}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
