"""Start the app: API and built UI on APP_PORT (default 5000).

Only one copy runs at a time: starting it again (a second click on start.bat) only opens the
browser. Press Ctrl+C to stop it, or run stop.bat / scripts/stop_app.py from anywhere.
"""

from __future__ import annotations

import argparse
import sys
import threading

from app import paths
from app.config import get_settings
from app.instance import InstanceLock, local_url, open_when_ready, read_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start AI Reel Maker.")
    parser.add_argument("--open-browser", action="store_true", help="open the app in the browser once it is ready")
    parser.add_argument("--self-test", action="store_true", help="check that this copy can run, without starting it")
    args = parser.parse_args(argv)
    if args.self_test:
        from app.selftest import run as self_test

        return self_test()

    paths.put_bundled_tools_first()
    settings = get_settings()
    settings.ensure_directories()
    lock = InstanceLock(settings.data_dir)
    if not lock.acquire():
        record = read_record(settings.data_dir) or {}
        port = int(record.get("port") or settings.app_port)
        print(f"AI Reel Maker is already running at {local_url(port)}")
        if args.open_browser and not open_when_ready(port):
            print("It did not answer yet. Open the address above in your browser in a moment.")
        return 0

    try:
        lock.write_record(settings.app_port)
        from app import create_app, socketio

        application = create_app(settings)
        if args.open_browser:
            threading.Thread(target=open_when_ready, args=(settings.app_port,), daemon=True).start()
        print(f"AI Reel Maker is running at {local_url(settings.app_port)}. Press Ctrl+C to stop.")
        socketio.run(
            application,
            host=settings.app_host,
            port=settings.app_port,
            allow_unsafe_werkzeug=True,
        )
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f"AI Reel Maker could not start on port {settings.app_port}: {exc}", file=sys.stderr)
        print("Another program may be using that port. Close it, or set APP_PORT in .env.", file=sys.stderr)
        return 1
    finally:
        lock.release()
    print("AI Reel Maker stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
