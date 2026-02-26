"""
Main application entry point for AI Agent Core System.
FastAPI application with API routes and lifecycle management.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings, setup_logging
from app.core.controller import Controller
from app.core.reasoning_core import ReasoningCore
from app.knowledge.database import Database, get_database
from app.knowledge.rag import RAG
from app.memory.long_term import LongTermMemory
from app.memory.vector_memory import VectorMemory
from app.models.schemas import (
    AgentRequest,
    AgentResponse,
    ErrorResponse,
    HealthCheckResponse,
    RAGQuery,
    RAGResponse,
)
from app.tools.examples import register_example_tools
from app.tools.router import ToolRouter, get_router

logger = logging.getLogger(__name__)

controller: Controller | None = None
database: Database | None = None
rag: RAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    global controller, database, rag
    
    logger.info("Starting AI Agent Core...")
    
    try:
        register_example_tools()
        logger.info("Example tools registered")
        
        database = Database()
        await database.initialize()
        logger.info("Database initialized")
        
        vector_memory = VectorMemory()
        await vector_memory.initialize()
        logger.info("Vector memory initialized")
        
        rag = RAG(vector_memory=vector_memory)
        logger.info("RAG initialized")
        
        long_term_memory = LongTermMemory()
        await long_term_memory.initialize()
        logger.info("Long-term memory initialized")
        
        tool_router = get_router()
        
        reasoning_core = ReasoningCore()
        
        controller = Controller(
            reasoning_core=reasoning_core,
            tool_router=tool_router,
            memory_manager=None
        )
        logger.info("Controller initialized")
        
        logger.info("AI Agent Core started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    finally:
        logger.info("Shutting down AI Agent Core...")
        
        if controller:
            await controller.shutdown()
        
        if rag:
            await rag.close()
        
        if database:
            await database.close()
        
        logger.info("AI Agent Core shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Pure capability AI Agent Core System",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="HTTPError",
            message=exc.detail
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="InternalServerError",
            message=str(exc) if settings.debug else "Internal server error"
        ).model_dump()
    )


@app.get("/health", response_model=HealthCheckResponse, tags=["System"])
async def health_check() -> HealthCheckResponse:
    """Check system health status."""
    components = {}
    
    if database:
        db_health = await database.health_check()
        components["database"] = "healthy" if db_health.get("healthy") else "unhealthy"
    else:
        components["database"] = "not_initialized"
    
    components["controller"] = "initialized" if controller else "not_initialized"
    components["rag"] = "initialized" if rag else "not_initialized"
    
    return HealthCheckResponse(
        status="healthy",
        version=settings.app_version,
        components=components
    )


@app.post("/agent/execute", response_model=AgentResponse, tags=["Agent"])
async def execute_agent(request: AgentRequest) -> AgentResponse:
    """
    Execute an agent request.
    
    This is the main entry point for agent execution.
    The agent will process the query and return a response.
    """
    if not controller:
        raise HTTPException(
            status_code=503,
            detail="Agent controller not initialized"
        )
    
    try:
        response = await controller.execute(request)
        return response
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/retrieve", response_model=RAGResponse, tags=["RAG"])
async def rag_retrieve(query: RAGQuery) -> RAGResponse:
    """
    Retrieve documents using RAG.
    
    Performs semantic search over indexed documents.
    """
    if not rag:
        raise HTTPException(
            status_code=503,
            detail="RAG not initialized"
        )
    
    try:
        response = await rag.retrieve_with_query(query)
        return response
    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/index", tags=["RAG"])
async def rag_index(
    content: str,
    session_id: str,
    metadata: dict = None
) -> dict:
    """
    Index a document for RAG retrieval.
    """
    if not rag:
        raise HTTPException(
            status_code=503,
            detail="RAG not initialized"
        )
    
    try:
        entry_id = await rag.index_document(
            content=content,
            session_id=session_id,
            metadata=metadata
        )
        return {"entry_id": entry_id, "status": "indexed"}
    except Exception as e:
        logger.error(f"RAG indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools", tags=["Tools"])
async def list_tools() -> dict:
    """List all available tools."""
    router = get_router()
    return {
        "tools": router.list_tools(),
        "count": len(router.list_tools())
    }


@app.get("/tools/{tool_name}", tags=["Tools"])
async def get_tool_schema(tool_name: str) -> dict:
    """Get schema for a specific tool."""
    router = get_router()
    schema = router.get_tool_schema(tool_name)
    
    if not schema:
        raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
    
    return schema


@app.get("/stats", tags=["System"])
async def get_stats() -> dict:
    """Get system statistics."""
    stats = {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment
    }
    
    if controller:
        stats["controller"] = {
            "sessions": len(controller._sessions)
        }
    
    router = get_router()
    stats["tools"] = router.get_stats()
    
    if rag:
        stats["rag"] = await rag.get_stats()
    
    return stats


@app.get("/", tags=["System"])
async def root() -> dict:
    """Root endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers
    )
