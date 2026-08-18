from app import create_app, socketio
from app.config import get_settings


def main() -> None:
    settings = get_settings()
    application = create_app(settings)
    socketio.run(
        application,
        host=settings.app_host,
        port=settings.app_port,
        allow_unsafe_werkzeug=True,
    )


if __name__ == "__main__":
    main()
