from __future__ import annotations

from urllib.parse import urlsplit

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from app.config import Settings, get_settings, set_settings
from app.db import init_db

# The Vite dev server (npm run dev). Every other page must come from this app itself.
DEV_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")


def origin_allowed(origin: str | None, environ: dict | None = None) -> bool:
    """Whether a browser page at ``origin`` may use the API. Requests without an Origin
    (curl, scripts, same-origin GETs) are allowed; a page on another site is not, so it cannot
    change settings or jobs on this local server."""
    if not origin:
        return True
    if origin in DEV_ORIGINS:
        return True
    host = urlsplit(origin).netloc.lower()
    environ = environ or {}
    own = {
        str(environ.get("HTTP_HOST") or "").lower(),
        str(environ.get("HTTP_X_FORWARDED_HOST") or "").split(",")[0].strip().lower(),
    }
    return bool(host) and host in own


socketio = SocketIO(async_mode="threading", cors_allowed_origins=origin_allowed)


def create_app(settings: Settings | None = None, start_background: bool = True) -> Flask:
    settings = settings or get_settings()
    set_settings(settings)
    settings.ensure_directories()

    app = Flask(__name__, static_folder=None)
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.config["SETTINGS"] = settings

    CORS(app, resources={r"/api/*": {"origins": list(DEV_ORIGINS)}})
    socketio.init_app(app, async_mode="threading", cors_allowed_origins=origin_allowed)

    @app.before_request
    def _same_origin_only():
        # CORS only hides responses; a page on another site could still send simple requests
        # (for example cancel or create a job). Changes must come from this app's own pages.
        if request.method in {"GET", "HEAD", "OPTIONS"}:
            return None
        if origin_allowed(request.headers.get("Origin"), request.environ):
            return None
        return jsonify({"error": "Requests from other websites are not allowed"}), 403
    init_db(settings.db_path)

    from app.routes.jobs import jobs_bp
    from app.routes.system import system_bp
    from app.routes.settings import settings_bp
    from app.routes.media import media_bp

    app.register_blueprint(jobs_bp, url_prefix="/api")
    app.register_blueprint(system_bp, url_prefix="/api")
    app.register_blueprint(settings_bp, url_prefix="/api")
    app.register_blueprint(media_bp, url_prefix="/api")

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def spa(path: str):
        if path.startswith("api/"):
            return {"error": "not found"}, 404
        dist = settings.frontend_dist
        candidate = dist / path
        if path and candidate.is_file():
            return send_from_directory(dist, path)
        index = dist / "index.html"
        if index.is_file():
            return send_from_directory(dist, "index.html")
        return {
            "name": "AI Video Highlights",
            "status": "frontend not built",
            "hint": "Run the Vite dev server or docker compose.",
        }, 200

    import app.services.job_manager as job_manager_mod

    job_manager_mod._manager = None

    if start_background:
        from app.services.ai_profile import schedule_startup_ai
        from app.workers.worker import start_worker

        schedule_startup_ai(settings)
        start_worker(app)

    return app
