"""Serve the read-only multi-run training dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import DEFAULT_WORKSPACE_ROOT
from .repository import DashboardRepository, RunNotArchivableError

STATIC_ROOT = Path(__file__).with_name("static")


def create_app(
    runs_root: Path = DEFAULT_WORKSPACE_ROOT / "runs",
    *,
    mlflow_uri: str | None = None,
) -> FastAPI:
    repository = DashboardRepository(runs_root)
    app = FastAPI(title="PalmClaw Memory Training", version="1.0")
    app.state.repository = repository
    app.state.mlflow_uri = mlflow_uri
    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "runs_root": str(repository.runs_root),
            "mlflow_uri": mlflow_uri,
            "read_only": False,
            "deletion_mode": "recoverable_archive",
        }

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        return repository.overview()

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> dict[str, Any]:
        try:
            return repository.detail(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown run") from exc

    @app.delete("/api/runs/{run_id}")
    def archive_run(
        run_id: str,
        confirm: Annotated[str, Query()],
    ) -> dict[str, Any]:
        if confirm != run_id:
            raise HTTPException(status_code=400, detail="Run confirmation mismatch")
        try:
            return repository.archive(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown run") from exc
        except RunNotArchivableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/comparison")
    def comparison(
        run_id: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, Any]:
        run_ids = run_id or []
        if len(run_ids) > 16:
            raise HTTPException(status_code=400, detail="At most 16 runs")
        try:
            return repository.comparison(run_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown run") from exc

    @app.get("/api/export")
    def export() -> dict[str, Any]:
        return repository.overview()

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root", type=Path, default=DEFAULT_WORKSPACE_ROOT / "runs"
    )
    parser.add_argument("--mlflow-uri")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5060)
    parser.add_argument("--export-summary", type=Path)
    return parser


def main() -> None:
    import uvicorn

    args = build_parser().parse_args()
    repository = DashboardRepository(args.runs_root)
    if args.export_summary:
        repository.export_summary(args.export_summary)
    uvicorn.run(
        create_app(args.runs_root, mlflow_uri=args.mlflow_uri),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
