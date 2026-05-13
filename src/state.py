from typing import List, Dict, Any, TypedDict, Annotated
import operator

class SubQueryState(TypedDict):
    query: str
    persona: str

class ResearchState(TypedDict):
    # Original input
    original_query: str
    
    # Phase 1: Planning
    clarified_query: str
    strategy: str
    sub_queries: List[SubQueryState]
    
    # Phase 2 & 3: Gathered Information
    # Using Annotated and operator.add to append new research data to the list
    gathered_data: Annotated[List[Dict[str, Any]], operator.add]
    
    # Phase 4: Reflection
    is_sufficient: bool
    reflection_notes: str
    feedback: str # Human feedback from the loop
    
    # Phase 5: Synthesis
    final_report: str
