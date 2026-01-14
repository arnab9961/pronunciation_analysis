import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from app.services.AI_voice_recognize.route import router as pronunciation_router


app = FastAPI(
    title="Ashlyn Prashad API",
    docs_url="/docs",
)


# Include routers
app.include_router(pronunciation_router)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8081)
   