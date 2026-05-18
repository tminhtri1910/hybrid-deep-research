# Hybrid Deep Research App

An agentic deep research application powered by LangGraph, LangChain, and FastAPI. It takes a research query, creates diverse personas/sub-queries, concurrently searches multiple sources (Web, YouTube, Reddit, TikTok, Local RAG), compresses findings, and pauses for human-in-the-loop reflection before synthesizing a final report.

## Quickstart

Follow these steps to get the backend running locally.

### 1. Set Up Environment

1. Ensure you have Python 3.9+ installed.
2. Clone the repository and navigate into the folder.
3. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   
   # Windows (PowerShell): 
   .\venv\Scripts\Activate.ps1
   # macOS/Linux:
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Configure API Keys

1. Copy the sample environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in your API keys:
   - `OPENAI_API_KEY`: Required for LLM routing and synthesis.
   - `OPENROUTER_API_KEY`: Optional, if you're using alternative models.
   - `TAVILY_API_KEY`: Required for general web searches.

### 3. Run the Server

Start the FastAPI backend using `uvicorn`:

```bash
uvicorn src.api:app --reload
```

The server will start at `http://127.0.0.1:8000`. You can explore the interactive API documentation at `http://127.0.0.1:8000/docs`.

### 4. Test the Pipeline

You can immediately test the pipeline using cURL (or the Swagger UI at `/docs`):

**Step 1: Start Research**
```bash
curl -X POST "http://127.0.0.1:8000/research" \
     -H "Content-Type: application/json" \
     -d '{"query": "Latest developments in solid state batteries"}'
```
*Take note of the `thread_id` returned.*

**Step 2: Check Status & Review Reflection**
```bash
curl -X GET "http://127.0.0.1:8000/research/<YOUR_THREAD_ID>"
```
*Wait until `next_node` is `synthesis_node`. This means the graph has paused for human review.*

**Step 3: Resume & Approve**
```bash
curl -X POST "http://127.0.0.1:8000/research/<YOUR_THREAD_ID>/resume" \
     -H "Content-Type: application/json" \
     -d '{"approve": true}'
```
*The graph will now generate your final report! Check the status again to view the `final_report` field.*
