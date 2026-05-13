from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.state import ResearchState
from src.nodes.planning import plan_research_node
from src.nodes.research import gather_research_node
from src.nodes.reflection import reflection_node
from src.nodes.synthesis import synthesis_node

def should_continue(state: ResearchState) -> Literal["synthesis_node", "plan_research_node"]:
    """
    Determine the next node based on reflection.
    """
    if state.get("is_sufficient"):
        return "synthesis_node"
    # If not sufficient, we loop back to planning.
    # The human feedback node will have injected `feedback` if the human intervened.
    return "plan_research_node"

def build_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("plan_research_node", plan_research_node)
    workflow.add_node("gather_research_node", gather_research_node)
    workflow.add_node("reflection_node", reflection_node)
    workflow.add_node("synthesis_node", synthesis_node)
    
    # Add edges
    workflow.add_edge(START, "plan_research_node")
    workflow.add_edge("plan_research_node", "gather_research_node")
    workflow.add_edge("gather_research_node", "reflection_node")
    
    # Conditional edge after reflection
    workflow.add_conditional_edges(
        "reflection_node",
        should_continue,
        {
            "synthesis_node": "synthesis_node",
            "plan_research_node": "plan_research_node"
        }
    )
    
    workflow.add_edge("synthesis_node", END)
    
    # Set up memory saver for state persistence (required for interrupt/human-in-the-loop)
    memory = MemorySaver()
    
    # We interrupt before synthesis so the human can review reflection and approve or provide feedback
    graph = workflow.compile(
        checkpointer=memory,
        interrupt_before=["synthesis_node"]
    )
    
    return graph

app_graph = build_graph()
