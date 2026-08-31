"""FastAPI 应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import hardware, deploy, monitor, target, store, tune, ai_tune

app = FastAPI(title="本地大模型部署助手", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(target.router, prefix="/api/target", tags=["目标机器"])
app.include_router(hardware.router, prefix="/api/hardware", tags=["硬件"])
app.include_router(deploy.router, prefix="/api/deploy", tags=["部署"])
app.include_router(monitor.router, prefix="/api/monitor", tags=["监控"])
app.include_router(store.router, prefix="/api/store", tags=["模型商店"])
app.include_router(tune.router, prefix="/api/tune", tags=["智能调优"])
app.include_router(ai_tune.router, prefix="/api/ai-tune", tags=["AI调优"])


@app.get("/")
def root():
    return {"name": "本地大模型部署助手", "version": "0.1.0"}
