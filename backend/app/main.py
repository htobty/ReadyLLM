"""FastAPI 应用入口"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api import hardware, deploy, monitor, target, store, tune, ai_tune
from .services.i18n import set_lang

app = FastAPI(title="本地大模型部署助手", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _lang_middleware(request: Request, call_next):
    """按 ?lang= 或 X-Lang 头设置界面语言，供后端提示消息按语言返回。

    同步路由跑在线程池里，anyio 会带上当前 context，所以 set_lang 在这里
    设置后，路由与 service 层取到的就是本次请求的语言。
    """
    set_lang(request.query_params.get("lang") or request.headers.get("x-lang") or "zh")
    return await call_next(request)

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
