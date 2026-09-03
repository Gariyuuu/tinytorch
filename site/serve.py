"""Local preview server mimicking the vercel.json routing.

Implements just enough of Vercel's behaviour to make `python site/serve.py` a
faithful preview: cleanUrls (/docs -> docs.html) and the /docs/:slug rewrite.

Usage:  python site/serve.py [port]
"""

from __future__ import annotations

import http.server
import os
import socketserver
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "public"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0].rstrip("/") or "/"

        # rewrite: /docs/:slug -> /docs.html
        parts = [p for p in clean.split("/") if p]
        if len(parts) == 2 and parts[0] == "docs":
            return str(ROOT / "docs.html")

        # cleanUrls: /docs -> /docs.html
        if clean != "/" and not Path(clean).suffix:
            candidate = ROOT / (clean.lstrip("/") + ".html")
            if candidate.is_file():
                return str(candidate)

        return super().translate_path(path)

    def log_message(self, fmt, *args):  # quieter output
        sys.stderr.write("  %s\n" % (fmt % args))


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 4173
    os.chdir(ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        print(f"TinyTorch docs preview → http://127.0.0.1:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
