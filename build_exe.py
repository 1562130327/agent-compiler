"""PyInstaller 打包脚本 — 生成单文件 EXE 桌面应用。

Usage:
    python build_exe.py

输出在 dist/ 目录下。
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# Ensure pyinstaller is installed
try:
    import PyInstaller  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

# Static file paths to include
static_dir = ROOT / "src" / "agent_compiler" / "web" / "static"
static_arg = f"--add-data={static_dir}{';'}static" if sys.platform == "win32" \
    else f"--add-data={static_dir}:static"

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--onefile",
    "--name", "AgentCompiler",
    "--console",
    "--clean",
    static_arg,
    "--hidden-import", "uvicorn.logging",
    "--hidden-import", "uvicorn.loops.auto",
    "--hidden-import", "uvicorn.protocols.http.auto",
    "--hidden-import", "fastapi",
    "--hidden-import", "agent_compiler",
    "--hidden-import", "agent_compiler.core",
    "--hidden-import", "agent_compiler.web",
    str(ROOT / "run_app.py"),
]

print("=" * 60)
print("  Building AgentCompiler.exe...")
print("=" * 60)
result = subprocess.run(cmd, cwd=str(ROOT))
if result.returncode == 0:
    print(f"\n  Build success! -> {ROOT / 'dist' / 'AgentCompiler.exe'}")
else:
    print(f"\n  Build failed with code {result.returncode}")
    sys.exit(result.returncode)
