from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import uvicorn
import numpy as np
from typing import List, Dict, Any, Literal
import json
from tool_detection import detect_tools

app = FastAPI(
    title="get tools api",
    description="API for video to tool",
    version="1.0.0"
)

class ToolRequest(BaseModel):
    image_urls: List[str] = Field(..., description="List of URLs of the screenshots to analyze")
    description: str = Field("", description="Optional additional context about the screenshots")
    
    @validator('image_urls')
    def validate_image_urls(cls, v):
        if not v:
            raise ValueError('At least one image URL must be provided')
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError('All image URLs must be valid HTTP or HTTPS URLs')
        return v

class ToolResponse(BaseModel):
    tools: str = Field(..., description="Comma-separated list of detected tools (Google Sheets, Gmail, Freshdesk)")


@app.post("/tool_names", response_model=ToolResponse)
async def tool_names(req: ToolRequest):
    """
    Detect tools being used in the provided screenshots.
    Returns a comma-separated list of detected tools (Google Sheets, Gmail, Freshdesk).
    """
    try:
        tools = detect_tools(req.image_urls, req.description)
        return ToolResponse(tools=tools)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Tool detection failed", "message": str(e)}
        )

    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)