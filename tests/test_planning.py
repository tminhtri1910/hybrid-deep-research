import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.state import ResearchState
from src.nodes.planning import plan_research_node

def test_plan_research_node():
    state: ResearchState = {
        "original_query": "What are the latest developments in solid state batteries?",
        "clarified_query": "",
        "strategy": "",
        "sub_queries": [],
        "gathered_data": [],
        "is_sufficient": False,
        "reflection_notes": "",
        "feedback": "",
        "final_report": ""
    }
    
    # Needs valid OPENAI_API_KEY in environment
    print("Testing plan_research_node...")
    result = plan_research_node(state)
    print("Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
        
if __name__ == "__main__":
    test_plan_research_node()
