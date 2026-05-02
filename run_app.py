"""EXE 启动入口 — 双击即启动 Web UI 桌面应用。

被 PyInstaller 打包为 AgentCompiler.exe 后，双击即可：
  1. 启动后台 FastAPI 服务
  2. 自动打开浏览器到 http://127.0.0.1:8220
  3. 关闭浏览器后 Ctrl+C 退出
"""

import sys
import os
import threading
import webbrowser
import time


def _find_static_dir() -> str:
    """Locate the static/ directory (handles PyInstaller _MEIPASS)."""
    if getattr(sys, 'frozen', False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    # Try PyInstaller-added-data path first
    static = os.path.join(base, "static")
    if os.path.isdir(static):
        return static
    # Fallback: source tree
    src_static = os.path.join(
        base, "src", "agent_compiler", "web", "static"
    )
    if os.path.isdir(src_static):
        return src_static
    return ""


def main():
    print("=" * 60)
    print("  Agent Compiler v0.3 — Desktop")
    print("=" * 60)
    print()

    # Override static dir if needed
    static_dir = _find_static_dir()
    if static_dir:
        from agent_compiler.web.server import STATIC_DIR as _STATIC
        import agent_compiler.web.server as srv
        srv.STATIC_DIR = type(srv.STATIC_DIR)(static_dir)

    from agent_compiler.web.server import create_app, app_web

    create_app()

    host = "127.0.0.1"
    port = 8220

    def open_browser():
        time.sleep(1.0)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    print(f"  浏览器即将打开: http://{host}:{port}")
    print(f"  按 Ctrl+C 退出")
    print()

    import uvicorn
    uvicorn.run(app_web, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
