import asyncio
from typing import Dict, Any, List
from src.state import ResearchState
from src.tools.search import search_general_web, search_reddit, search_youtube, search_tiktok, search_local_documents

async def gather_research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node to execute concurrent searches based on the selected strategy and sub-queries.
    """
    sub_queries = state.get("sub_queries", [])
    strategy = state.get("strategy", "general_web")
    
    gathered_data = []
    
    # We will gather tasks to run concurrently
    tasks = []
    
    for sq in sub_queries:
        query = sq["query"]
        persona = sq["persona"]
        
        # Decide which tools to run based on strategy
        if strategy == "social_trending":
            tasks.append(search_reddit.ainvoke({"query": query}))
            tasks.append(search_tiktok.ainvoke({"query": query}))
            tasks.append(search_youtube.ainvoke({"query": query}))
        elif strategy == "academic":
            tasks.append(search_general_web.ainvoke({"query": query + " filetype:pdf scholarly"}))
            tasks.append(search_local_documents.ainvoke({"query": query}))
        else: # general_web
            tasks.append(search_general_web.ainvoke({"query": query}))
            tasks.append(search_youtube.ainvoke({"query": query}))
    
    # Run all search tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for res in results:
        if isinstance(res, Exception):
            print(f"Task failed: {res}")
            continue
        if isinstance(res, list):
            for item in res:
                if isinstance(item, dict):
                    gathered_data.append(item)
    
    # Return updates to state (using operator.add semantics, this will append)
    return {"gathered_data": gathered_data}
