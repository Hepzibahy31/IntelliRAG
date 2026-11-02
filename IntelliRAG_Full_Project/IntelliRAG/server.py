"""IntelliRAG web server (FastAPI). Run:  python server.py  ->  http://localhost:8000"""
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from rag import config, llm
from rag.answer import ask, options_from_dict
from rag.documents import SUPPORTED, load_bytes
from rag.evaluation import judge, load_eval_set, run_benchmark
from rag.vector_store import KnowledgeBase, ingest_folder

kb = KnowledgeBase()
STATE = {"phase": "starting", "message": "Starting..."}
HITS = defaultdict(deque)
ASK_LIMIT = int(config.get("ASK_LIMIT_PER_MIN", "30"))
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _startup_work():
    try:
        STATE.update(phase="indexing", message="Loading saved index...")
        kb.load()
        STATE.update(phase="indexing", message="Indexing documents in the data folder (first run takes a few minutes)...")
        ingest_folder(kb, log=lambda m: STATE.update(message=m))
        STATE.update(phase="ready", message="Ready")
    except Exception as error:
        STATE.update(phase="ready", message=f"Started with a problem: {error}")


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=_startup_work, daemon=True).start()
    yield


app = FastAPI(title="IntelliRAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class AskRequest(BaseModel):
    question: str
    history: list = []
    options: dict = {}


class JudgeRequest(BaseModel):
    question: str
    answer: str
    context: str


def _rate_limit(request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")
    now, q = time.time(), HITS[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= ASK_LIMIT:
        raise HTTPException(429, "Too many questions. Please wait a minute and try again.")
    q.append(now)


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    return {"phase": STATE["phase"], "message": STATE["message"], "llm": llm.info(),
            "documents": len(kb.documents()), "chunks": kb.size(),
            "eval_questions": len(load_eval_set())}


@app.get("/api/documents")
def documents():
    return {"documents": kb.documents()}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    report = []
    for f in files:
        name = Path(f.filename or "file").name
        if Path(name).suffix.lower() not in SUPPORTED:
            report.append({"name": name, "error": "Only PDF, TXT and MD files are supported."})
            continue
        data = await f.read()
        if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
            report.append({"name": name, "error": f"File is larger than {config.MAX_UPLOAD_MB} MB."})
            continue
        if name in kb.sources():
            report.append({"name": name, "error": "Already indexed."})
            continue
        try:
            docs = load_bytes(name, data)
            if not docs:
                report.append({"name": name, "error": "No readable text found (scanned PDF?)."})
                continue
            added = await run_in_threadpool(kb.add_documents, docs)
            report.append({"name": name, "chunks": added})
        except Exception as error:
            report.append({"name": name, "error": str(error)[:200]})
    return {"report": report, "documents": kb.documents()}


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    if not kb.remove_document(name):
        raise HTTPException(404, "Document not found.")
    return {"documents": kb.documents()}


@app.post("/api/ask")
def ask_endpoint(req: AskRequest, request: Request):
    question = req.question.strip()
    if not question:
        raise HTTPException(400, "Please type a question.")
    if len(question) > 1000:
        raise HTTPException(400, "Question is too long (max 1000 characters).")
    if kb.size() == 0:
        msg = ("Documents are still being indexed. Please wait a moment." if STATE["phase"] != "ready"
               else "No documents yet. Upload a PDF or TXT file first.")
        raise HTTPException(503, msg)
    _rate_limit(request)
    history = [m for m in req.history[-6:] if isinstance(m, dict) and "role" in m and "content" in m]
    return ask(kb, question, history, options_from_dict(req.options))


@app.post("/api/judge")
def judge_endpoint(req: JudgeRequest, request: Request):
    _rate_limit(request)
    try:
        return judge(req.question, req.answer, req.context)
    except llm.LLMError as error:
        raise HTTPException(400, str(error))


@app.post("/api/evaluate")
def evaluate_endpoint(request: Request):
    if kb.size() == 0:
        raise HTTPException(503, "No documents indexed yet.")
    _rate_limit(request)
    return run_benchmark(kb)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
