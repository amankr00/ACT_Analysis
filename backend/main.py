import os
import re
import shutil
import time
import threading
import traceback
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

try:
    from ollama import Client
    ollama_client = Client(host=OLLAMA_HOST)
except Exception as e:
    ollama_client = None
    print(f"Ollama client initialization failed. Is Ollama running at {OLLAMA_HOST}?", e)


def _strip_think(text: str) -> str:
    """Removes qwen3 <think>...</think> reasoning blocks that may leak into the output."""
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Handle an unterminated opening tag defensively.
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def _ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Calls the local Ollama model and returns the cleaned text content."""
    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        think=False,  # qwen3 is a reasoning model; disable chain-of-thought output
        options={"temperature": temperature, "num_predict": max_tokens},
    )
    return _strip_think(response["message"]["content"])

RELEVANCE_THRESHOLD = 0.50  # Cosine similarity threshold for "relevant" chunks

def verify_citations(answer: str, retrieved_chunks: list) -> str:
    """Verifies that any section/rule numbers cited by the AI actually exist in the retrieved chunks."""
    pattern = re.compile(r'(?i)\b((?:Section|Rule)\s+\d+[a-zA-Z]*(?:\([a-zA-Z0-9]+\))?)\b')
    matches = pattern.findall(answer)
    
    if not matches:
        return answer
        
    chunks_text = " ".join([c.get('text', '') for c in retrieved_chunks]).lower()
    unique_matches = set(matches)
    unverified = []
    
    for match in unique_matches:
        num_search = re.search(r'\d+', match)
        if num_search:
            core_num = num_search.group()
            if not re.search(rf'\b{core_num}\b', chunks_text):
                answer = answer.replace(match, f"~~{match}~~ ⚠️")
                unverified.append(match)
                
    if unverified:
        answer += "\n\n---\n⚠️ **Citation Notice:** The following references were NOT found in your database and may be inaccurate: " + ", ".join(f"`{u}`" for u in sorted(unverified)) + ". Please verify independently."
        
    return answer


def extract_issue_queries(statement: str) -> List[str]:
    """Uses LLM to extract multiple distinct legal issue queries from a user's statement for multi-query retrieval."""
    if not ollama_client:
        return [statement]
    try:
        content = _ollama_chat(
            messages=[
                {"role": "system", "content": """You are a legal issue extractor. Given a user's legal situation, identify the distinct legal issues and output each as a short search query on a separate line.

Rules:
- Output exactly 3 to 5 queries, one per line.
- Each query should be 3-8 words of specific legal terminology.
- No numbering, no bullets, no explanation.
- Each query should target a different legal concept.

Example input: "A landless agricultural laborer has lived in a homestead on the landlord's land for 20 years and faces forcible eviction."
Example output:
homestead rights landless laborer
protection from forcible eviction
parcha issuance settlement rights
long-term possession agricultural land
land reform homestead tenancy"""},
                {"role": "user", "content": statement}
            ],
            temperature=0.0,
            max_tokens=150,
        )
        queries = [q.strip() for q in content.strip().split("\n") if q.strip()]
        return queries[:5] if queries else [statement]
    except Exception:
        return [statement]


def filter_by_relevance(results: List[Dict], threshold: float = RELEVANCE_THRESHOLD) -> tuple:
    """Splits results into high-relevance and low-relevance based on FAISS cosine similarity score."""
    high_relevance = []
    low_relevance = []
    for r in results:
        if r.get("faiss_score", 0.0) >= threshold:
            high_relevance.append(r)
        else:
            low_relevance.append(r)
    return high_relevance, low_relevance


def compute_confidence(high_relevance: List[Dict], all_results: List[Dict]) -> str:
    """Returns HIGH, MEDIUM, or LOW confidence based on how many chunks passed the relevance threshold."""
    if not all_results:
        return "LOW"
    ratio = len(high_relevance) / len(all_results)
    avg_score = sum(r.get("faiss_score", 0) for r in high_relevance) / max(len(high_relevance), 1)
    if ratio >= 0.4 and avg_score >= 0.55:
        return "HIGH"
    elif ratio >= 0.2 or avg_score >= 0.45:
        return "MEDIUM"
    return "LOW"


def synthesize_answer(statement: str, high_chunks: List[Dict], low_chunks: List[Dict], confidence: str) -> str:
    if not ollama_client:
        return f"Ollama is not available. Ensure Ollama is running at {OLLAMA_HOST} and the '{OLLAMA_MODEL}' model is pulled to enable AI synthesis."
    
    if not high_chunks and not low_chunks:
        return "No clauses were found in the database for this statement."
        
    try:
        # Build context: act name + clause text for fact-matching
        context_parts = []
        for i, c in enumerate(high_chunks + low_chunks, 1):
            act = c.get('act_name', 'Unknown Act')
            text = c.get('text', '')
            score = c.get('faiss_score', 0)
            context_parts.append(f"[Source {i}] Act: {act} | Score: {score:.2f}\n{text}")
        context = "\n\n---\n\n".join(context_parts)

        system_message = """You are a Bihar Legal RAG system that retrieves Acts, Rules, Circulars, and Case Laws from a FAISS vector database.

Your objective is to maximize relevance and minimize hallucinations.

STRICT REQUIREMENTS:

1. NEVER search for an exact factual match to the user's statement.
   Instead:
   - Identify the underlying legal issue(s).
   - Return the laws that govern those legal issues.

2. DO NOT summarize retrieved clauses or explain retrieval reasoning.

3. DO NOT output:
   - Confidence, Match Type, Ranking, Search Process, Legal Issue Extraction
   - Notes, Disclaimers
   - "No direct match found" or "Retrieved clauses are unrelated"
   - Mentions of laws that are NOT applicable
   - Any chain-of-thought reasoning

4. HALLUCINATION PREVENTION RULES:
   - Before returning a law, verify that it exists in the retrieved database OR is a well-established Indian/Bihar statute.
   - If uncertain whether a law exists, exclude it entirely.
   - Never create laws by combining legal concepts into Act names.
   - OUTPUT ONLY VERIFIED ACTS.

5. ABSOLUTE SILENCE ON REJECTIONS (CRITICAL):
   - If a law is deemed inapplicable or fails verification, DO NOT print it.
   - DO NOT print "(Removed as per Hallucination Prevention Rule)".
   - DO NOT print "(Omitted)".
   - DO NOT leave blank bullet points.
   - Simply pretend the rejected law never existed. The user must only see the valid, applicable laws.

6. If a section number is not explicitly present in retrieved materials or is uncertain, do NOT mention the section number. Prefer Act names over section citations.

7. Return a maximum of 5 laws.

8. Each law must contain only:
   - Law Name
   - One short support statement explaining its relevance to the user's statement (1–2 lines)

OUTPUT FORMAT:

### Relevant Bihar Laws

• [Exact Law Name]
  Support: [One short affirmative sentence explaining why this Act is relevant to the user's specific statement.]

• [Exact Law Name]
  Support: [One short affirmative sentence explaining why this Act is relevant to the user's specific statement.]

RULES FOR OUTPUT:
- You must output ONLY the "Relevant Bihar Laws" header and the bulleted list.
- You must ONLY include laws that ARE applicable.
- Do NOT output any preamble, postamble, or chain-of-thought text.
- Maximum 5 laws. Minimum 1 law.
- Return nothing except the laws above."""

        user_prompt = f"""**User's Statement:**
"{statement}"

**Retrieved Clauses from Database:**

{context}

Return the most relevant Bihar laws. Adhere strictly to the OUTPUT FORMAT. Do not include any reasoning or mention inapplicable laws. If a law is not relevant, do not mention it at all."""

        raw_answer = _ollama_chat(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=512,
        )
        
        # Post-Processing: Forcefully remove any lines where the LLM hallucinated a "Removed" note
        cleaned_lines = []
        for line in raw_answer.split("\n"):
            lower_line = line.lower()
            if "(removed" not in lower_line and "(omitted" not in lower_line and "not applicable" not in lower_line:
                cleaned_lines.append(line)
                
        cleaned_answer = "\n".join(cleaned_lines).strip()
        
        final_answer = verify_citations(cleaned_answer, high_chunks + low_chunks)
        print("\n=== GENERATED AI RESPONSE ===")
        print(final_answer)
        print("=============================\n")
        return final_answer
    except Exception as e:
        return f"Could not generate an answer via Ollama. Error: {str(e)}"
from core.embedder import embed_query
from core.reranker import rerank as rerank_chunks
from core.faiss_act_store import load_index as load_act_index
from core.faiss_act_store import search as act_search
from core.faiss_act_store import get_total_chunks as get_act_total_chunks
from act.ingest import ingest_acts

app = FastAPI(
    title="Judicial AI Acts Backend",
    description="Vector search and ingestion API for statutory Acts and laws",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://actanalysis.netlify.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_private_network_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


logger = logging.getLogger(__name__)

class ActSearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ActSearchQueryRequest(BaseModel):
    statement: Optional[str] = ""

def _clean_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()

def _normalize_top_k(value: Any) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        raise ValueError("top_k must be an integer.")
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20.")
    return top_k

def _ok(response: Any, message: str) -> dict:
    return {"status": "success", "message": message, "response": response}

def _json_error(message: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"status": "error", "message": message},
    )

# All public endpoints are served under this prefix (matches the Cloudflare tunnel route).
router = APIRouter(prefix="/bihar-act")

# === Async job store for long-running searches (polling pattern) ===
# Keeps the request alive past the Cloudflare tunnel / proxy ~100s limit:
# the client submits a job, then polls the status endpoint until it completes.
JOB_TTL_SECONDS = 15 * 60  # jobs are retained for 15 minutes
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

def _purge_expired_jobs() -> None:
    """Removes jobs older than JOB_TTL_SECONDS to keep the in-memory store bounded."""
    now = time.time()
    with _jobs_lock:
        expired = [jid for jid, j in _jobs.items() if now - j.get("created_at", now) > JOB_TTL_SECONDS]
        for jid in expired:
            _jobs.pop(jid, None)

def _run_act_search_job(job_id: str, combined_query: str) -> None:
    """Background worker: runs the full act search and stores the result on the job."""
    try:
        result = _perform_act_search(combined_query)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="completed", result=result, updated_at=time.time())
    except Exception as e:
        traceback.print_exc()
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id].update(status="error", message=str(e), updated_at=time.time())

@app.on_event("startup")
def startup():
    print("Loading ACTS FAISS index...")
    try:
        load_act_index()
        print(f"ACTS FAISS ready | Chunks: {get_act_total_chunks()}")
    except Exception as e:
        print(f"Error loading ACTS FAISS on startup: {str(e)}")

@app.get("/")
def root():
    return {"message": "Judicial AI Acts Backend is running"}

@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "database": "ok",
    }


def _perform_act_search(combined_query: str) -> dict:
    """Runs the full two-stage act search + answer synthesis. Returns the response payload.

    Raises on failure; callers (the background job runner) are responsible for
    capturing the exception and surfacing it via the job status.
    """
    load_act_index()

    # === PASS 1: Search with original statement ===
    query_embedding = embed_query(combined_query)
    top_k = 15
    results = act_search(query_embedding, k=top_k, include_scores=True)

    formatted_results = []
    for item in results:
        metadata = item.get("metadata", {})
        formatted_results.append({
            "text": item.get("text", ""),
            "act_name": metadata.get("act_name", "Statutory Act"),
            "document_name": metadata.get("document_name", "N/A"),
            "filename": metadata.get("filename", "N/A"),
            "pdf_path": metadata.get("pdf_path", "N/A"),
            "page_num": metadata.get("page_num", "N/A"),
            "chunk_index": metadata.get("chunk_index", "N/A"),
            "faiss_score": round(float(item.get("vector_score", 0.0)), 4)
        })

    # === RELEVANCE FILTERING ===
    high_relevance, low_relevance = filter_by_relevance(formatted_results)
    confidence = compute_confidence(high_relevance, formatted_results)

    # === STAGE 2: If low confidence, extract multiple legal issue queries and search again ===
    issue_queries = None
    if confidence == "LOW" and ollama_client:
        issue_queries = extract_issue_queries(combined_query)
        print(f"[STAGE 2] Issue queries: {issue_queries}")

        existing_texts = {r["text"] for r in formatted_results}
        for iq in issue_queries:
            query_embedding_iq = embed_query(iq)
            results_iq = act_search(query_embedding_iq, k=10, include_scores=True)

            for item in results_iq:
                text = item.get("text", "")
                if text not in existing_texts:
                    metadata = item.get("metadata", {})
                    new_result = {
                        "text": text,
                        "act_name": metadata.get("act_name", "Statutory Act"),
                        "document_name": metadata.get("document_name", "N/A"),
                        "filename": metadata.get("filename", "N/A"),
                        "pdf_path": metadata.get("pdf_path", "N/A"),
                        "page_num": metadata.get("page_num", "N/A"),
                        "chunk_index": metadata.get("chunk_index", "N/A"),
                        "faiss_score": round(float(item.get("vector_score", 0.0)), 4)
                    }
                    formatted_results.append(new_result)
                    existing_texts.add(text)

        # Re-filter after multi-query merge
        high_relevance, low_relevance = filter_by_relevance(formatted_results)
        confidence = compute_confidence(high_relevance, formatted_results)

    # === SYNTHESIZE ANSWER ===
    generated_answer = synthesize_answer(combined_query, high_relevance, low_relevance, confidence)

    return {
        "query": combined_query,
        "issue_queries": issue_queries,
        "top_k": top_k,
        "confidence": confidence,
        "high_relevance_count": len(high_relevance),
        "low_relevance_count": len(low_relevance),
        "results": formatted_results,
        "generated_answer": generated_answer
    }


@router.post("/act/search")
async def start_act_search(request: ActSearchQueryRequest):
    """Submits a search job and returns a job_id immediately.

    The client then polls GET /bihar-act/act/search/status/{job_id} until the
    job completes. This keeps total latency from tripping the ~100s proxy limit.
    """
    combined_query = request.statement.strip() if request.statement else ""
    if not combined_query:
        return _json_error("Provide a statement for search query.", status_code=400)

    _purge_expired_jobs()
    job_id = str(uuid.uuid4())
    now = time.time()
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "created_at": now, "updated_at": now}

    threading.Thread(target=_run_act_search_job, args=(job_id, combined_query), daemon=True).start()

    return {
        "status": "pending",
        "job_id": job_id,
        "poll_url": f"/bihar-act/act/search/status/{job_id}",
        "message": "Search started. Poll the status endpoint for results.",
    }


@router.get("/act/search/status/{job_id}")
async def get_act_search_status(job_id: str):
    """Returns the current status of a search job, including its result once completed."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        job = dict(job) if job else None

    if not job:
        return _json_error("Job not found or expired.", status_code=404)

    if job["status"] == "completed":
        return _ok(job["result"], "Acts search completed successfully.")
    if job["status"] == "error":
        return _json_error(job.get("message", "Search failed."))

    # Still running.
    return {"status": "pending", "job_id": job_id, "message": "Search in progress."}


@router.post("/search")
async def search_acts(request: ActSearchRequest):
    try:
        query = _clean_text(request.query)
        if not query:
            raise ValueError("Query cannot be empty.")

        top_k = _normalize_top_k(request.top_k)
        load_act_index()
        query_embedding = embed_query(query)
        results = act_search(query_embedding, k=top_k, include_scores=True)

        formatted_results = []
        for item in results:
            metadata = item.get("metadata", {})
            formatted_results.append({
                "text": item.get("text", ""),
                "act_name": metadata.get("act_name", "Statutory Act"),
                "document_name": metadata.get("document_name", "N/A"),
                "filename": metadata.get("filename", "N/A"),
                "pdf_path": metadata.get("pdf_path", "N/A"),
                "page_num": metadata.get("page_num", "N/A"),
                "chunk_index": metadata.get("chunk_index", "N/A"),
                "faiss_score": round(float(item.get("vector_score", 0.0)), 4)
            })

        return _ok(
            {
                "query": query,
                "top_k": top_k,
                "result_count": len(formatted_results),
                "results": formatted_results
            },
            "Acts search completed successfully."
        )
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc))

@router.post("/ingest")
async def trigger_ingestion():
    """
    Trigger ingestion of all PDF files in the data/output_batch directory.
    Runs ingest_acts() and returns the summary result.
    """
    try:
        result = ingest_acts()
        if result.get("status") == "error":
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": result.get("message", "Ingestion failed.")}
            )
        return _ok(result, f"Ingestion complete. Processed {result.get('processed_pdfs', 0)} PDFs, {result.get('total_chunks', 0)} chunks indexed.")
    except Exception as e:
        traceback.print_exc()
        return _json_error(str(e))


# Register all /bihar-act endpoints on the app.
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "5090"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
