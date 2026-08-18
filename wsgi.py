from app import create_app, socketio
from app.config import get_settings

settings = get_settings()
app = create_app(settings)

if __name__ == "__main__":
    socketio.run(app, host=settings.app_host, port=settings.app_port, allow_unsafe_werkzeug=True)
