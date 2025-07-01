from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import traceback
from typing import Dict, Any

from app.agent import AgentWrapper, long_term_mem

app = FastAPI(title="AI Assistant API")

agent = AgentWrapper()  # <-- inisialisasi wrapper di sini

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query) -> Dict[str, Any]:
    try:
        # Process the query
        print(f"Received query: {query.text}")
        result = await agent.ainvoke(query.text)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        # Store in memory
        long_term_mem.add(query.text, result["output"])
        
        return {
            "status": "success",
            "input": query.text,
            "output": result["output"],
            "intermediate_steps": result.get("intermediate_steps", [])
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }
        )

@app.get("/memory")
async def get_memory() -> Dict[str, Any]:
    try:
        history = long_term_mem.load()
        return {
            "status": "success",
            "history": history
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": str(e)
            }
        )