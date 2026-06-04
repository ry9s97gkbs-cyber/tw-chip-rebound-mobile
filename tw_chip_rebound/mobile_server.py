"""Mobile-friendly web server for the public-data screener."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import argparse
import socket
import traceback
from urllib.parse import parse_qs, urlparse

from .public_fetcher import FinMindError, screen_with_public_data
from .public_fetcher import clean_token


ROOT = Path(__file__).resolve().parent.parent
APP_HTML = ROOT / "outputs" / "mobile_app.html"
SERVER_TOKEN = clean_token(os.environ.get("FINMIND_TOKEN", ""))


class MobileHandler(BaseHTTPRequestHandler):
    def _send_headers(self, status: int, content_type: str, length: int = 0) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(length))
        self.end_headers()

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_headers(status, "application/json; charset=utf-8", len(body))
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = APP_HTML.read_bytes()
        self._send_headers(200, "text/html; charset=utf-8", len(body))
        self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler name.
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/app"):
            self._send_headers(200, "text/html; charset=utf-8", APP_HTML.stat().st_size)
            return
        if parsed.path in ("/api/ping", "/api/config"):
            self._send_headers(200, "application/json; charset=utf-8", 0)
            return
        self._send_headers(404, "application/json; charset=utf-8", 0)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler name.
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/app"):
            self._send_html()
            return
        if parsed.path == "/api/ping":
            self._send_json(200, {"ok": True})
            return
        if parsed.path == "/api/config":
            self._send_json(200, {"has_server_token": bool(SERVER_TOKEN)})
            return
        if parsed.path != "/api/screen":
            self._send_json(404, {"error": "not found"})
            return

        params = parse_qs(parsed.query)
        token = clean_token(params.get("token", [""])[0]) or SERVER_TOKEN
        target_date = params.get("date", [""])[0] or None
        days = int(params.get("days", ["45"])[0])
        branch_limit = int(params.get("branch_limit", ["80"])[0])
        if not token:
            self._send_json(400, {"error": "請先輸入 FinMind token。"})
            return
        try:
            result = screen_with_public_data(
                token=token,
                target_date=target_date,
                days=days,
                branch_limit=branch_limit,
            )
        except FinMindError as exc:
            self._send_json(502, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"rows": result.rows, "meta": result.meta})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name.
        parsed = urlparse(self.path)
        if parsed.path != "/api/screen":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            payload = json.loads(body)
        except (ValueError, json.JSONDecodeError):
            self._send_json(400, {"error": "請求格式錯誤。"})
            return

        token = clean_token(str(payload.get("token", ""))) or SERVER_TOKEN
        target_date = str(payload.get("date") or "") or None
        days = int(payload.get("days") or 45)
        branch_limit = int(payload.get("branch_limit") or 80)
        if not token:
            self._send_json(400, {"error": "請先輸入 FinMind token。"})
            return
        try:
            result = screen_with_public_data(
                token=token,
                target_date=target_date,
                days=days,
                branch_limit=branch_limit,
            )
        except FinMindError as exc:
            self._send_json(502, {"error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_json(500, {"error": str(exc)})
            return
        self._send_json(200, {"rows": result.rows, "meta": result.meta})

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"[mobile] {self.address_string()} {fmt % args}")


def run(host: str = "0.0.0.0", port: int = 8787) -> None:
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), MobileHandler)
    urls = [f"http://localhost:{port}"]
    try:
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if not ip.startswith("127."):
                urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    print("Mobile screener ready:")
    for url in dict.fromkeys(urls):
        print(f"  {url}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="台股價弱籌碼強手機版伺服器")
    parser.add_argument("--host", default="0.0.0.0", help="預設 0.0.0.0，讓同 Wi-Fi 手機可連線")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8787")), help="預設 8787 或 Render PORT")
    args = parser.parse_args()
    run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
