"""FastAPI backend for the MoE inspector GUI."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .session import Inspector

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="MoE Inspector")
inspector = Inspector()


class LoadRequest(BaseModel):
    model_id: str
    device: str = "auto"
    dtype: str = "auto"


class MaskRequest(BaseModel):
    disabled: dict[str, list[int]] | None = None
    sleeping: list[int] | None = None
    mode: str | None = None


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    temperature: float = 0.0
    chat_template: bool = True
    compare: bool = False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
def status() -> dict:
    return inspector.status()


@app.post("/api/load")
def load(request: LoadRequest) -> dict:
    inspector.load(request.model_id, request.device, request.dtype)
    return inspector.status()


@app.post("/api/mask")
def mask(request: MaskRequest) -> dict:
    return inspector.set_mask(request.disabled, request.sleeping, request.mode)


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict:
    try:
        result = inspector.generate(
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            chat_template=request.chat_template,
            ablate=True,
        )
        if request.compare:
            result["baseline"] = inspector.generate(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                chat_template=request.chat_template,
                ablate=False,
            )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="MoE Inspector")
    parser.add_argument("--model", help="model id to load on startup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.model:
        inspector.load(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
