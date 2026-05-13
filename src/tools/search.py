import os
import json
import httpx
from typing import List, Dict, Any
from langchain_core.tools import tool
import yt_dlp

@tool
async def search_general_web(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search the general web using SearXNG or Tavily.
    """
    searxng_url = os.getenv("SEARXNG_URL", "http://localhost:8080")
    tavily_api = os.getenv("TAVILY_API_KEY")
    
    if tavily_api:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": tavily_api, "query": query, "max_results": max_results}
            )
            if response.status_code == 200:
                data = response.json()
                return [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")} for r in data.get("results", [])]
                
    return [{"title": "Mock Web Result", "url": "https://example.com", "snippet": f"Web result for {query}"}]

@tool
async def search_reddit(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search Reddit for discussions using public JSON API.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    async with httpx.AsyncClient() as client:
        url = f"https://www.reddit.com/search.json?q={query}&limit={limit}"
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            return [
                {
                    "title": post["data"].get("title"),
                    "url": f"https://reddit.com{post['data'].get('permalink')}",
                    "content": post["data"].get("selftext", "")[:500]
                }
                for post in posts
            ]
    return []

@tool
def search_youtube(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """
    Search YouTube and extract metadata using yt-dlp.
    """
    ydl_opts = {
        'quiet': True,
        'extract_flat': True,
        'force_generic_extractor': False,
    }
    results = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            if 'entries' in info:
                for entry in info['entries']:
                    results.append({
                        "title": entry.get('title'),
                        "url": entry.get('url'),
                        "description": entry.get('description', '')[:200],
                        "channel": entry.get('uploader')
                    })
        except Exception as e:
            print(f"yt-dlp error: {e}")
    return results

@tool
async def search_tiktok(query: str) -> List[Dict[str, Any]]:
    """
    Search TikTok using public wrapper or mock.
    """
    return [{"title": "Mock TikTok", "url": f"https://tiktok.com/search?q={query}", "description": f"TikTok about {query}"}]

@tool
def search_local_documents(query: str, doc_dir: str = "data/") -> List[Dict[str, Any]]:
    """
    Search local documents using LlamaIndex SimpleDirectoryReader and SummaryIndex.
    """
    try:
        from llama_index.core import SimpleDirectoryReader, SummaryIndex
        from llama_index.core.response.schema import Response
        
        if not os.path.exists(doc_dir):
            return [{"error": f"Directory {doc_dir} does not exist."}]
            
        documents = SimpleDirectoryReader(doc_dir).load_data()
        if not documents:
            return [{"error": "No documents found."}]
            
        index = SummaryIndex.from_documents(documents)
        query_engine = index.as_query_engine()
        response: Response = query_engine.query(query)
        
        return [{"source_nodes": len(response.source_nodes), "response": str(response)}]
    except ImportError:
        return [{"error": "llama-index is not installed."}]
    except Exception as e:
        return [{"error": str(e)}]
