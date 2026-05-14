from __future__ import annotations

import threading
import uuid
from datetime import date
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_core.messages import HumanMessage, SystemMessage

from blog_agent import blog_graph, llm

app = FastAPI(title="Blog Writer AI")
templates = Jinja2Templates(directory="templates")

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")

# ─────────────────────────────────────────────────────────────
# In-memory job store
# ─────────────────────────────────────────────────────────────

jobs: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────

def _run_generation(job_id: str, topic: str) -> None:
    try:
        result = blog_graph.invoke({
            "topic": topic,
            "as_of": date.today().isoformat(),
        })
        plan = result.get("plan")
        jobs[job_id].update({
            "status": "ready",
            "final": result.get("final", ""),
            "plan_title": plan.blog_title if plan else topic,
        })
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})


REWRITE_SYSTEM = (
    "You are a senior technical editor. Revise the blog post based on the feedback provided. "
    "Preserve the overall structure but improve it according to the feedback. "
    "Return the full revised blog in Markdown, starting with the # title."
)


def _run_rewrite(job_id: str, feedback: str) -> None:
    try:
        current = jobs[job_id]["final"]
        revised = llm.invoke([
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(content=f"ORIGINAL BLOG:\n{current}\n\nFEEDBACK:\n{feedback}\n\nRevised blog:"),
        ]).content.strip()
        jobs[job_id].update({
            "status": "ready",
            "final": revised,
            "revision_count": jobs[job_id].get("revision_count", 0) + 1,
        })
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate(request: Request, topic: str = Form(...)):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "generating",
        "topic": topic,
        "final": "",
        "plan_title": topic,
        "revision_count": 0,
    }
    threading.Thread(target=_run_generation, args=(job_id, topic), daemon=True).start()
    return RedirectResponse(f"/generating/{job_id}", status_code=303)


@app.get("/generating/{job_id}", response_class=HTMLResponse)
async def generating_page(request: Request, job_id: str):
    job = jobs.get(job_id)
    if not job:
        return RedirectResponse("/")
    return templates.TemplateResponse("generating.html", {
        "request": request,
        "job_id": job_id,
        "topic": job["topic"],
        "is_rewriting": job.get("revision_count", 0) > 0,
        "revision_count": job.get("revision_count", 0),
    })


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"status": "not_found"}
    return {
        "status": job["status"],
        "error": job.get("error"),
        "revision_count": job.get("revision_count", 0),
    }


@app.get("/review/{job_id}", response_class=HTMLResponse)
async def review_page(request: Request, job_id: str):
    job = jobs.get(job_id)
    if not job:
        return RedirectResponse("/")
    if job["status"] not in ("ready", "approved"):
        return RedirectResponse(f"/generating/{job_id}")
    return templates.TemplateResponse("review.html", {
        "request": request,
        "job_id": job_id,
        "topic": job["topic"],
        "plan_title": job["plan_title"],
        "content": job["final"],
        "revision_count": job.get("revision_count", 0),
        "max_revisions": 3,
    })


@app.post("/reject/{job_id}")
async def reject_blog(job_id: str, feedback: str = Form(...)):
    job = jobs.get(job_id)
    if not job:
        return RedirectResponse("/", status_code=303)
    if job.get("revision_count", 0) >= 3:
        return RedirectResponse(f"/review/{job_id}", status_code=303)
    jobs[job_id]["status"] = "rewriting"
    jobs[job_id]["feedback"] = feedback
    threading.Thread(target=_run_rewrite, args=(job_id, feedback), daemon=True).start()
    return RedirectResponse(f"/generating/{job_id}", status_code=303)


@app.post("/approve/{job_id}")
async def approve_blog(job_id: str):
    if job_id not in jobs:
        return RedirectResponse("/", status_code=303)
    jobs[job_id]["status"] = "approved"
    return RedirectResponse(f"/success/{job_id}", status_code=303)


@app.get("/success/{job_id}", response_class=HTMLResponse)
async def success_page(request: Request, job_id: str):
    job = jobs.get(job_id, {})
    return templates.TemplateResponse("success.html", {
        "request": request,
        "job_id": job_id,
        "topic": job.get("topic", ""),
        "plan_title": job.get("plan_title", ""),
        "revision_count": job.get("revision_count", 0),
        "content": job.get("final", ""),
    })
