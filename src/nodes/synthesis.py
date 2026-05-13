from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.state import ResearchState
from src.llm import get_llm

def synthesis_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node to synthesize all gathered data into a final comprehensive markdown report.
    """
    original_query = state.get("original_query", "")
    gathered_data = state.get("gathered_data", [])
    
    # Format data for the prompt
    sources_text = ""
    for idx, d in enumerate(gathered_data):
        title = d.get("title", "Unknown Title")
        url = d.get("url", "Unknown URL")
        content = d.get("content", d.get("snippet", d.get("description", "")))
        sources_text += f"Source [{idx+1}]: {title}\nURL: {url}\nContent: {content}\n\n"
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert researcher and technical writer. Synthesize the provided sources into a highly detailed, well-structured markdown report that directly answers the user's query.\n"
                   "You MUST include inline citations in the format [Source ID] to reference the sources used. At the end of the report, include a 'References' section listing the sources and their URLs."),
        ("user", "Original Query: {query}\n\nSources:\n{sources}")
    ])
    
    llm = get_llm(provider="openai", model="gpt-4o", temperature=0.3)
    chain = prompt | llm
    
    result = chain.invoke({
        "query": original_query,
        "sources": sources_text
    })
    
    return {
        "final_report": result.content
    }
