from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
from typing import Dict, Any, Optional

from src.graph import app_graph
from langchain_core.runnables import RunnableConfig

app = FastAPI(title="Hybrid Deep Research API")

class ResearchRequest(BaseModel):
    query: str

class ResumeRequest(BaseModel):
    approve: bool
    feedback: Optional[str] = None

@app.post("/research")
async def start_research(request: ResearchRequest):
    """
    Start a new research thread.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "original_query": request.query,
        "gathered_data": []
    }
    
    # We run the graph asynchronously
    # It will run until it hits the interrupt_before synthesis_node, or finishes.
    app_graph.invoke(initial_state, config=config)
    
    return {"thread_id": thread_id, "status": "started"}

@app.get("/research/{thread_id}")
async def get_state(thread_id: str):
    """
    Get the current state of a research thread.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = app_graph.get_state(config)
    
    if not state_snapshot:
        raise HTTPException(status_code=404, detail="Thread not found")
        
    next_node = state_snapshot.next
    values = state_snapshot.values
    
    return {
        "thread_id": thread_id,
        "next_node": next_node,
        "is_sufficient": values.get("is_sufficient"),
        "reflection_notes": values.get("reflection_notes"),
        "gathered_data_count": len(values.get("gathered_data", [])),
        "final_report": values.get("final_report")
    }

@app.post("/research/{thread_id}/resume")
async def resume_research(thread_id: str, request: ResumeRequest):
    """
    Resume research. If approve=True, synthesis happens.
    If approve=False, feedback is injected and graph routes back to planning.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = app_graph.get_state(config)
    
    if not state_snapshot or not state_snapshot.next:
        raise HTTPException(status_code=400, detail="Thread not found or not waiting")
        
    if request.approve:
        # Proceed with synthesis
        app_graph.invoke(None, config=config)
    else:
        # Update state with feedback and override is_sufficient to force loop back
        app_graph.update_state(
            config,
            {"feedback": request.feedback, "is_sufficient": False},
            as_node="reflection_node" # Act as if reflection node outputted this
        )
        # Proceed
        app_graph.invoke(None, config=config)
        
    return {"status": "resumed"}
