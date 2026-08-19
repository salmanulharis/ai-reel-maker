"""Device performance baseline and post-optimization benchmark."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def bench_import_app() -> dict:
    t0 = time.perf_counter()
    from app import create_app  # noqa: WPS433

    t_import = time.perf_counter()
    from app.config import get_settings

    settings = get_settings()
    app = create_app(settings)
    t_ready = time.perf_counter()
    return {
        "import_create_app_ms": round((t_import - t0) * 1000, 1),
        "create_app_ms": round((t_ready - t_import) * 1000, 1),
        "total_startup_ms": round((t_ready - t0) * 1000, 1),
    }


def bench_http(label: str, url: str, client) -> dict:
    times = []
    status = None
    for _ in range(5):
        t0 = time.perf_counter()
        resp = client.get(url)
        times.append(_ms(t0))
        status = resp.status_code
    return {
        "label": label,
        "url": url,
        "status": status,
        "avg_ms": round(statistics.mean(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
    }


def bench_ollama_text() -> dict:
    import requests

    payload = {
        "model": "qwen3:8b",
        "messages": [{"role": "user", "content": 'Reply JSON only: {"ok":true}'}],
        "stream": False,
        "think": False,
        "format": "json",
        "keep_alive": "15m",
        "options": {"num_ctx": 4096, "num_predict": 64, "temperature": 0.1, "num_thread": 6},
    }
    runs = []
    for label in ("cold_or_first", "warm_second"):
        t0 = time.perf_counter()
        resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=600)
        elapsed = _ms(t0)
        body = resp.json() if resp.ok else {}
        eval_count = body.get("eval_count")
        eval_duration = body.get("eval_duration")
        tps = None
        if eval_count and eval_duration:
            tps = round(eval_count / (eval_duration / 1e9), 2)
        runs.append(
            {
                "label": label,
                "status": resp.status_code,
                "latency_ms": elapsed,
                "eval_count": eval_count,
                "tokens_per_sec": tps,
                "load_duration_ms": round((body.get("load_duration") or 0) / 1e6, 1),
            }
        )
    return {"runs": runs, "latency_ms": runs[-1]["latency_ms"], "load_duration_ms": runs[-1]["load_duration_ms"]}


def main() -> None:
    out: dict = {"ts": time.time()}
    out["startup"] = bench_import_app()

    try:
        from app import create_app
        from app.config import get_settings

        app = create_app(get_settings())
        client = app.test_client()
        out["http"] = [
            bench_http("home", "/", client),
            bench_http("ollama_status", "/api/system/ollama/status", client),
            bench_http("jobs_list", "/api/jobs", client),
        ]
    except Exception as exc:  # noqa: BLE001
        out["http_error"] = str(exc)

    try:
        out["ollama_text"] = bench_ollama_text()
    except Exception as exc:  # noqa: BLE001
        out["ollama_text_error"] = str(exc)

    path = ROOT / "data" / "perf_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and "startup" in existing and "after" not in existing:
            out = {"before": existing, "after": out}
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
