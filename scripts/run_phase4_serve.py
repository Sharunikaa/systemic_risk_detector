"""
scripts/run_phase4_serve.py

Phase 4 entry point — Start both API and frontend dev server.

Starts:
  1. FastAPI server on http://localhost:8000  (uvicorn)
  2. React Vite dev server on http://localhost:5173

Both run as subprocesses. Ctrl+C terminates both cleanly.
"""

import subprocess
import sys
import time
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_FRONTEND = _ROOT / "frontend"


def main():
    print("=" * 60)
    print("QML Contagion Detection System — Phase 4 Serve")
    print("=" * 60)
    print(f"API:       http://localhost:8000")
    print(f"Dashboard: http://localhost:5173")
    print(f"Health:    http://localhost:8000/api/health")
    print("=" * 60)
    print("Press Ctrl+C to stop both servers.")
    print()

    # ── Start FastAPI ────────────────────────────────────────────
    api_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ],
        cwd=str(_ROOT),
    )

    # Give FastAPI a moment to start
    time.sleep(3)

    # ── Start Vite dev server ─────────────────────────────────────
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(_FRONTEND),
    )

    try:
        api_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        api_proc.terminate()
        frontend_proc.terminate()
        api_proc.wait(timeout=5)
        frontend_proc.wait(timeout=5)
        print("Done.")


if __name__ == "__main__":
    sys.path.insert(0, str(_ROOT))
    main()
