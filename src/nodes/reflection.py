from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.state import ResearchState
from src.llm import get_llm

class ReflectionOutput(BaseModel):
    is_sufficient: bool = Field(description="True if the gathered data is sufficient to answer the original query comprehensively, False otherwise.")
    reflection_notes: str = Field(description="Notes on what is missing or what should be searched next if insufficient, or a summary of why it is sufficient.")

def reflection_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node to reflect on the gathered data and determine if more research is needed.
    """
    original_query = state.get("original_query", "")
    gathered_data = state.get("gathered_data", [])
    
    # We summarize the data slightly to avoid huge context, or just pass titles
    data_summary = "\n".join([f"- {d.get('title', 'Unknown')}: {str(d.get('snippet', d.get('description', d.get('content', ''))))[:200]}" for d in gathered_data])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research supervisor. Review the gathered data against the original query.\n"
                   "Determine if the information is sufficient to write a comprehensive report.\n"
                   "If yes, set is_sufficient to true. If no, set it to false and provide reflection_notes on what is missing."),
        ("user", "Original Query: {query}\n\nGathered Data:\n{data}")
    ])
    
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(ReflectionOutput)
    
    chain = prompt | structured_llm
    
    result: ReflectionOutput = chain.invoke({
        "query": original_query,
        "data": data_summary
    })
    
    return {
        "is_sufficient": result.is_sufficient,
        "reflection_notes": result.reflection_notes
    }
