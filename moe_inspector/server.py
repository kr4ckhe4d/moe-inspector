"""FastAPI backend for the MoE inspector GUI."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import strata

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="MoE Inspector")
STRATA_DIRS: list[Path] = list(strata.DEFAULT_DIRS)


class _LazyInspector:
    """The live-model Inspector needs torch; the /strata view does not. Import it on first use."""

    _real = None

    def __getattr__(self, name):
        if _LazyInspector._real is None:
            from .session import Inspector
            _LazyInspector._real = Inspector()
        return getattr(_LazyInspector._real, name)


inspector = _LazyInspector()


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
    try:
        return inspector.status()
    except ImportError as exc:      # a light install: the /strata view works, the live one needs torch
        return {"state": "error", "message": f"live view needs torch ({exc.name} missing): pip install -e .",
                "layers": []}


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


@app.get("/strata")
def strata_page() -> FileResponse:
    return FileResponse(STATIC / "strata.html")


def _strata_file(name: str, kind: str) -> Path:
    for f in strata.list_files(STRATA_DIRS)[kind]:
        if f["name"] == name:
            return Path(f["path"])
    raise HTTPException(status_code=404, detail=f"no {kind[:-1]} named {name!r}")


@app.get("/api/strata/files")
def strata_files() -> dict:
    return strata.list_files(STRATA_DIRS) | {"dirs": [str(d) for d in STRATA_DIRS]}


@lru_cache(maxsize=8)
def _trace(path: str, mtime: float) -> dict:
    return strata.trace_payload(Path(path))


@app.get("/api/strata/trace")
def strata_trace(name: str) -> dict:
    p = _strata_file(name, "traces")
    return _trace(str(p), p.stat().st_mtime)


@app.get("/api/strata/profile")
def strata_profile(name: str) -> dict:
    return strata.read_profile(_strata_file(name, "profiles"))


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="MoE Inspector")
    parser.add_argument("--model", help="model id to load on startup")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--strata-dir", action="append", type=Path, default=None,
                        help="a folder of Strata routing traces and profiles (*.bin) for /strata; repeatable")
    args = parser.parse_args()

    if args.strata_dir:
        STRATA_DIRS[:] = args.strata_dir
    if args.model:
        inspector.load(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
