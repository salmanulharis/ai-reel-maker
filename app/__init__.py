from __future__ import annotations

from flask import Flask, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from app.config import Settings, get_settings, set_settings
from app.db import init_db

socketio = SocketIO(async_mode="threading", cors_allowed_origins="*")


def create_app(settings: Settings | None = None, start_background: bool = True) -> Flask:
    settings = settings or get_settings()
    set_settings(settings)
    settings.ensure_directories()

    app = Flask(__name__, static_folder=None)
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_bytes
    app.config["SETTINGS"] = settings

    CORS(app, resources={r"/api/*": {"origins": "*"}})
    socketio.init_app(app, async_mode="threading", cors_allowed_origins="*")
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
        from app.workers.worker import start_worker

        start_worker(app)

    return app
