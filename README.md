# 🤖 Blog Writing Agent App

An autonomous, multi-agent blog writing pipeline built with **LangGraph**, **LangChain**, and **Ollama**. Give it a topic — it researches, plans, writes, and generates a complete, image-ready Markdown blog post automatically.

---

## ✨ Features

- **Intelligent Routing** — Automatically decides if web research is needed (`closed_book` / `hybrid` / `open_book` modes)
- **Live Web Research** — Fetches and synthesizes up-to-date evidence via Tavily Search API
- **Structured Planning** — Generates a detailed multi-section blog outline with goals, bullets, and word targets
- **Parallel Writing** — Spawns concurrent worker agents per section using LangGraph's `Send` API (fanout pattern)
- **AI Image Generation** — Decides where images improve understanding and generates them via Gemini 2.5 Flash
- **Multiple Blog Formats** — Supports `explainer`, `tutorial`, `news_roundup`, `comparison`, `system_design`
- **Full Markdown Output** — Saves a ready-to-publish `.md` file with embedded images

---

## 🏗️ Architecture

```
Topic Input
    │
    ▼
[Router] ──── closed_book ────────────────────┐
    │                                          │
    └── needs_research ──► [Research]          │
                               │               │
                               ▼               ▼
                         [Orchestrator] (Plan Generator)
                               │
                    ┌──────────┼──────────┐
                    ▼          ▼          ▼
                [Worker]   [Worker]   [Worker]   (parallel fanout)
                    └──────────┼──────────┘
                               ▼
                          [Reducer]
                         /    |    \
                [Merge] → [Images] → [Output]
                               │
                               ▼
                        final_blog.md
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph (StateGraph + Send API) |
| LLM | Ollama (llama3.1 default) |
| Web Research | Tavily Search API + LangChain |
| Image Generation | Google Gemini 2.5 Flash |
| Structured Outputs | Pydantic + LangChain `.with_structured_output()` |
| Backend Notebook | Jupyter (`bwa_backend.ipynb`) |
| Frontend | HTML (Jinja2 templates) |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally with `llama3.1` pulled
- API keys for Tavily (optional) and Google Gemini (optional)

### Installation

```bash
git clone https://github.com/gurpreet-singh-ji/Blog-Writing-Agent-App.git
cd Blog-Writing-Agent-App
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:

```env
OLLAMA_MODEL=llama3.1         # or any Ollama model
TAVILY_API_KEY=your_key       # optional: enables web research
GOOGLE_API_KEY=your_key       # optional: enables AI image generation
```

### Run

```bash
python main.py
```

Or explore the pipeline interactively in `bwa_backend.ipynb`.

---

## 📁 Project Structure

```
Blog-Writing-Agent-App/
├── blog_agent.py        # Core LangGraph pipeline (router, research, orchestrator, workers, reducer)
├── main.py              # Entry point
├── bwa_backend.ipynb    # Interactive Jupyter notebook
├── templates/           # HTML frontend templates
├── images/              # AI-generated images (auto-created at runtime)
└── .env                 # API keys (not committed)
```

---

## 🔄 How It Works

1. **Router** — Analyzes the topic and decides the research mode
2. **Research** (if needed) — Runs parallel Tavily queries, deduplicates, filters by recency
3. **Orchestrator** — Creates a structured `Plan` with 5–9 tasks, each with goals, bullets, and word targets
4. **Workers** — Each section is written in parallel by an independent worker agent
5. **Reducer** — Merges sections, decides image placement, generates images via Gemini, saves final `.md`

---

## 📝 Example Output

Input:
```
Topic: "How to deploy LLMs in production"
```

Output: A fully structured Markdown blog post with sections like Introduction, Architecture, Deployment Strategies, Monitoring, and a Conclusion — with inline AI-generated diagrams.

---

## 🤝 Contributing

PRs welcome! Feel free to open issues for bugs or feature requests.

---

## 📄 License

MIT License

---

Built by [Gurpreet Singh](https://github.com/gurpreet-singh-ji) • MLOps & LLMOps Engineer
