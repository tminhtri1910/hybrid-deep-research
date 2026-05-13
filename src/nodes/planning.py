from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.state import ResearchState
from src.schemas import PlanningOutput
from src.llm import get_llm

def plan_research_node(state: ResearchState) -> Dict[str, Any]:
    """
    Node to clarify the query, select a strategy, and generate sub-queries via personas.
    """
    original_query = state.get("original_query", "")
    feedback = state.get("feedback", "")
    
    # Optional feedback integration
    feedback_context = f"\nHuman Feedback from previous loop: {feedback}" if feedback else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Your task is to plan a deep research operation.\n"
                   "1. Clarify and expand the original query to ensure it is unambiguous.\n"
                   "2. Select a strategy ('academic', 'social_trending', 'general_web').\n"
                   "3. Generate 3-5 distinct sub-queries, each from a unique persona/perspective (e.g., skeptic, supporter, analyst) to ensure comprehensive coverage."),
        ("user", "Original Query: {query}{feedback_context}")
    ])
    
    # We use a structured output LLM (OpenAI API supports this)
    llm = get_llm(provider="openai", model="gpt-4o-mini", temperature=0.2)
    structured_llm = llm.with_structured_output(PlanningOutput)
    
    chain = prompt | structured_llm
    
    result: PlanningOutput = chain.invoke({
        "query": original_query,
        "feedback_context": feedback_context
    })
    
    # Return updates to the state
    return {
        "clarified_query": result.clarified_query,
        "strategy": result.strategy,
        "sub_queries": [{"query": sq.query, "persona": sq.persona} for sq in result.sub_queries]
    }
