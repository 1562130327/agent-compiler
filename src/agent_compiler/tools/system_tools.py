"""Real system tools — shell execution, file ops, web, git, processes.

These are actual implementations, not mocks. They work on the real system.
Dangerous operations (shell, write, delete) require confirmation by default.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

# ── Safety config ──────────────────────────────────────────────────────

SHELL_TIMEOUT = 60  # seconds
SHELL_MAX_OUTPUT = 100_000  # chars
DANGEROUS_COMMANDS = [
    "rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:",  # fork bomb
    "shutdown", "reboot", "halt", "poweroff",
    "chmod 777 /", "chown -R", "> /dev/sda",
]


def _is_dangerous(command: str) -> str | None:
    """Check if a shell command looks dangerous. Returns reason or None."""
    cmd_lower = command.lower().strip()
    for d in DANGEROUS_COMMANDS:
        if d in cmd_lower:
            return f"命令包含危险操作: {d}"
    return None


# ── Shell execution ────────────────────────────────────────────────────

def execute_shell(command: str, workdir: str | None = None,
                  timeout: int | None = None) -> dict:
    """Execute a shell command and return stdout, stderr, and exit code.

    Safety: dangerous commands are blocked. Timeout prevents runaway processes.
    """
    danger = _is_dangerous(command)
    if danger:
        return {"command": command, "success": False, "exit_code": -1,
                "stdout": "", "stderr": danger, "blocked": True}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout or SHELL_TIMEOUT,
            cwd=workdir or os.getcwd(),
            encoding="utf-8",
            errors="replace",
        )
        stdout = result.stdout[:SHELL_MAX_OUTPUT]
        stderr = result.stderr[:SHELL_MAX_OUTPUT]
        truncated = (len(result.stdout) > SHELL_MAX_OUTPUT or
                    len(result.stderr) > SHELL_MAX_OUTPUT)
        return {
            "command": command,
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": truncated,
        }
    except subprocess.TimeoutExpired:
        return {"command": command, "success": False, "exit_code": -1,
                "stdout": "", "stderr": f"命令超时 ({timeout or SHELL_TIMEOUT}s)"}
    except Exception as e:
        return {"command": command, "success": False, "exit_code": -1,
                "stdout": "", "stderr": str(e)}


# ── File operations ────────────────────────────────────────────────────

def read_file(path: str, lines: int | None = None,
              offset: int = 0) -> dict:
    """Read a file and return its contents."""
    p = Path(path).resolve()
    if not p.exists():
        return {"path": str(p), "success": False, "error": "文件不存在"}
    if p.is_dir():
        return {"path": str(p), "success": False, "error": "路径是目录不是文件"}
    try:
        size = p.stat().st_size
        # Limit to 100KB by default for LLM context
        content = p.read_text(encoding="utf-8", errors="replace")
        all_lines = content.splitlines()
        if offset > 0:
            all_lines = all_lines[offset:]
        if lines is not None:
            all_lines = all_lines[:lines]
        truncated = len(all_lines) < len(content.splitlines())
        return {
            "path": str(p),
            "success": True,
            "content": "\n".join(all_lines),
            "size_bytes": size,
            "total_lines": len(content.splitlines()),
            "returned_lines": len(all_lines),
            "truncated": truncated,
        }
    except Exception as e:
        return {"path": str(p), "success": False, "error": str(e)}


def write_file(path: str, content: str, append: bool = False) -> dict:
    """Write content to a file. Creates parent directories if needed."""
    p = Path(path).resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(p, mode, encoding="utf-8") as f:
            f.write(content)
        return {
            "path": str(p),
            "success": True,
            "bytes_written": len(content.encode("utf-8")),
            "mode": "append" if append else "overwrite",
        }
    except Exception as e:
        return {"path": str(p), "success": False, "error": str(e)}


def list_directory(path: str = ".", limit: int = 50,
                   pattern: str | None = None) -> dict:
    """List files and directories at a path (real implementation)."""
    p = Path(path).resolve()
    if not p.exists():
        return {"path": str(p), "success": False, "error": "目录不存在"}
    if not p.is_dir():
        return {"path": str(p), "success": False, "error": "路径不是目录"}
    try:
        items = []
        for entry in sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
            if pattern and not fnmatch(entry.name, pattern):
                continue
            if len(items) >= limit:
                break
            try:
                stat = entry.stat()
            except OSError:
                stat = None
            items.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size_bytes": stat.st_size if stat and entry.is_file() else None,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat() if stat else "",
            })
        return {"path": str(p), "total": len(items), "files": items}
    except Exception as e:
        return {"path": str(p), "success": False, "error": str(e)}


def search_files(directory: str, pattern: str,
                 file_glob: str = "*", max_results: int = 50) -> dict:
    """Search for a regex pattern within files in a directory. Like grep -r."""
    import fnmatch as _fnmatch
    p = Path(directory).resolve()
    if not p.exists():
        return {"directory": str(p), "success": False, "error": "目录不存在"}

    matches = []
    try:
        for fpath in p.rglob(file_glob):
            if not fpath.is_file():
                continue
            if fpath.stat().st_size > 5 * 1024 * 1024:  # skip >5MB files
                continue
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                try:
                    if re.search(pattern, line, re.IGNORECASE):
                        matches.append({
                            "file": str(fpath.relative_to(p)),
                            "line": i,
                            "content": line.strip()[:200],
                        })
                        if len(matches) >= max_results:
                            break
                except re.error:
                    break
            if len(matches) >= max_results:
                break
        return {
            "directory": str(p),
            "pattern": pattern,
            "file_glob": file_glob,
            "total_matches": len(matches),
            "matches": matches,
            "truncated": len(matches) >= max_results,
        }
    except Exception as e:
        return {"directory": str(p), "success": False, "error": str(e)}


# ── Web tools ──────────────────────────────────────────────────────────

def web_fetch(url: str, max_chars: int = 10_000) -> dict:
    """Fetch content from a URL and return as text."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AgentCompiler/0.2)",
                "Accept": "text/html,text/plain,*/*",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()
            # Try to decode
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()
            try:
                text = raw.decode(charset, errors="replace")
            except Exception:
                text = raw.decode("utf-8", errors="replace")

            # Strip HTML tags for readability
            if "text/html" in content_type:
                text = _strip_html(text)

            text = text[:max_chars]
            return {
                "url": url,
                "success": True,
                "content": text,
                "content_type": content_type,
                "status_code": resp.status,
                "truncated": len(raw) > max_chars,
            }
    except urllib.error.HTTPError as e:
        return {"url": url, "success": False,
                "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"url": url, "success": False, "error": str(e)}


def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web and return results (title, URL, snippet).

    Uses DuckDuckGo's HTML search (no API key required).
    """
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; AgentCompiler/0.2)",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        results = _parse_duckduckgo_html(html)[:max_results]
        return {
            "query": query,
            "total_results": len(results),
            "results": results,
        }
    except Exception as e:
        return {"query": query, "success": False, "error": str(e)}


# ── Git tools ──────────────────────────────────────────────────────────

def _run_git(args: list[str], workdir: str | None = None) -> dict:
    """Run a git command and return results."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=workdir or os.getcwd(),
            encoding="utf-8",
            errors="replace",
        )
        return {
            "command": f"git {' '.join(args)}",
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as e:
        return {"command": f"git {' '.join(args)}",
                "exit_code": -1, "stdout": "", "stderr": str(e)}


def git_status(path: str | None = None) -> dict:
    """Show git working tree status."""
    r = _run_git(["status", "--short"], path)
    files = [line.strip() for line in r["stdout"].split("\n") if line.strip()]
    return {
        "path": path or os.getcwd(),
        "files": files,
        "file_count": len(files),
        "exit_code": r["exit_code"],
        "error": r["stderr"] if r["exit_code"] != 0 else None,
    }


def git_diff(path: str | None = None, staged: bool = False) -> dict:
    """Show git diff of working tree changes."""
    args = ["diff"]
    if staged:
        args.append("--staged")
    r = _run_git(args + ["--stat"], path)
    full_diff = _run_git(args, path)
    return {
        "path": path or os.getcwd(),
        "stat": r["stdout"][:2000] if r["exit_code"] == 0 else "",
        "diff": full_diff["stdout"][:5000],
        "exit_code": r["exit_code"],
        "error": r["stderr"] if r["exit_code"] != 0 else None,
    }


def git_log(path: str | None = None, max_count: int = 10) -> dict:
    """Show git commit log."""
    r = _run_git([
        "log", f"--max-count={max_count}",
        "--oneline", "--decorate", "--date=short", "--pretty=format:%h %ad %s"
    ], path)
    commits = [line.strip() for line in r["stdout"].split("\n") if line.strip()]
    return {
        "path": path or os.getcwd(),
        "commits": commits,
        "count": len(commits),
        "exit_code": r["exit_code"],
        "error": r["stderr"] if r["exit_code"] != 0 else None,
    }


# ── Process management ─────────────────────────────────────────────────

def list_processes(sort_by: str = "cpu", limit: int = 20) -> dict:
    """List running processes, sorted by CPU or memory usage (cross-platform)."""
    processes = []
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["tasklist", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            for line in result.stdout.strip().split("\n"):
                parts = line.strip('"').split('","')
                if len(parts) >= 5:
                    try:
                        processes.append({
                            "name": parts[0],
                            "pid": int(parts[1]),
                            "memory_kb": int(parts[4].replace(" K", "").replace(",", "")),
                        })
                    except ValueError:
                        pass
            processes.sort(key=lambda p: -p["memory_kb"])
        else:
            result = subprocess.run(
                ["ps", "aux", "--no-headers"],
                capture_output=True, text=True, timeout=10,
                encoding="utf-8", errors="replace",
            )
            for line in result.stdout.strip().split("\n"):
                cols = line.split()
                if len(cols) >= 11:
                    try:
                        processes.append({
                            "user": cols[0],
                            "pid": int(cols[1]),
                            "cpu_pct": float(cols[2]),
                            "mem_pct": float(cols[3]),
                            "command": " ".join(cols[10:]),
                        })
                    except ValueError:
                        pass
            if sort_by == "mem":
                processes.sort(key=lambda p: -p["mem_pct"])
            else:
                processes.sort(key=lambda p: -p["cpu_pct"])
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {"total": len(processes), "processes": processes[:limit],
            "sort_by": sort_by, "platform": sys.platform}


# ── Helpers ────────────────────────────────────────────────────────────

def _strip_html(html: str) -> str:
    """Remove HTML tags and extract text content."""
    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text: list[str] = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "noscript"):
                self._skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "noscript"):
                self._skip = False
            if tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "h5", "tr"):
                self.text.append("\n")

        def handle_data(self, data):
            if not self._skip:
                self.text.append(data.strip())

    s = _Stripper()
    s.feed(html)
    raw = " ".join(t for t in s.text if t)
    # Collapse whitespace
    return re.sub(r"\s+", " ", raw).strip()


def _parse_duckduckgo_html(html: str) -> list[dict]:
    """Extract search results from DuckDuckGo HTML."""
    results = []
    # Each result is in a div with class "result"
    # Title is in <a class="result__a">
    # URL is in the href of that <a>
    # Snippet is in <a class="result__snippet">
    title_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    titles = title_pattern.findall(html)
    snippets = [re.sub(r"<[^>]+>", "", s).strip() for s in snippet_pattern.findall(html)]

    for i, (href, title) in enumerate(titles):
        title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = snippets[i] if i < len(snippets) else ""
        if title and href.startswith("http"):
            results.append({
                "title": title,
                "url": href,
                "snippet": snippet[:300],
            })
    return results


def fnmatch(name: str, pattern: str) -> bool:
    """Simple glob matching for file names."""
    import fnmatch as _fnmatch
    return _fnmatch.fnmatch(name, pattern)


# ── Code editing ──────────────────────────────────────────────────────

def edit_file(path: str, old_string: str, new_string: str) -> dict:
    """Replace old_string with new_string in a file (exact match, once).

    This is the primary code editing tool. It performs a single, precise
    text replacement. The old_string must match exactly (including whitespace).
    If old_string appears multiple times, only the first is replaced.
    """
    p = Path(path).resolve()
    if not p.exists():
        return {"path": str(p), "success": False, "error": "文件不存在"}
    try:
        content = p.read_text(encoding="utf-8")
        if old_string not in content:
            return {
                "path": str(p),
                "success": False,
                "error": "未找到要替换的内容（请确保 old_string 完全匹配，包括空格和缩进）",
                "hint": "使用 read_file 查看文件内容，确认 old_string 精确匹配",
            }
        count = content.count(old_string)
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return {
            "path": str(p),
            "success": True,
            "occurrences": count,
            "replaced": 1,
            "bytes_before": len(content),
            "bytes_after": len(new_content),
        }
    except Exception as e:
        return {"path": str(p), "success": False, "error": str(e)}


def glob_files(pattern: str, path: str = ".") -> dict:
    """Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts')."""
    from glob import iglob
    p = Path(path).resolve()
    try:
        matches = []
        for fpath in iglob(str(p / pattern), recursive=True):
            f = Path(fpath)
            if f.is_file():
                try:
                    stat = f.stat()
                except OSError:
                    stat = None
                matches.append({
                    "path": str(f.relative_to(p)),
                    "size_bytes": stat.st_size if stat else 0,
                })
        matches.sort(key=lambda x: x["path"])
        return {
            "pattern": pattern,
            "base_path": str(p),
            "total": len(matches),
            "files": matches[:100],
            "truncated": len(matches) > 100,
        }
    except Exception as e:
        return {"pattern": pattern, "success": False, "error": str(e)}


# ── Programming tools ──────────────────────────────────────────────────

def execute_python(code: str, workdir: str | None = None,
                    timeout: int = 30) -> dict:
    """Execute Python code in a subprocess and return stdout, stderr, and return code.

    The code runs in an isolated subprocess. Variables persist within the code block.
    Use this for calculations, data processing, file manipulation, or running scripts.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir or os.getcwd(),
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        stdout = result.stdout[:SHELL_MAX_OUTPUT]
        stderr = result.stderr[:SHELL_MAX_OUTPUT]
        return {
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeout_sec": timeout,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "exit_code": -1,
                "stdout": "", "stderr": f"代码执行超时 ({timeout}s)"}
    except Exception as e:
        return {"success": False, "exit_code": -1,
                "stdout": "", "stderr": str(e)}


def install_package(package: str, upgrade: bool = False) -> dict:
    """Install a Python package using pip. Returns install output."""
    args = [sys.executable, "-m", "pip", "install", "--quiet"]
    if upgrade:
        args.append("--upgrade")
    args.append(package)
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
        return {
            "package": package,
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "stdout": result.stdout.strip()[-500:],
            "stderr": result.stderr.strip()[-500:],
        }
    except Exception as e:
        return {"package": package, "success": False, "error": str(e)}


def run_tests(path: str = ".", test_type: str = "pytest",
               extra_args: str = "") -> dict:
    """Run tests in a directory. Supports pytest, unittest, or custom command."""
    commands = {
        "pytest": [sys.executable, "-m", "pytest", path, "-v"],
        "unittest": [sys.executable, "-m", "unittest", "discover", path],
    }
    cmd = commands.get(test_type, [sys.executable, "-m", "pytest", path, "-v"])
    if extra_args:
        cmd.extend(extra_args.split())

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=path,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout + result.stderr)[:5000]
        # Count pass/fail
        passed = len(re.findall(r"\b(PASSED|PASS)\b", output))
        failed = len(re.findall(r"\b(FAILED|FAIL)\b", output))
        return {
            "path": path,
            "test_type": test_type,
            "success": result.returncode == 0,
            "exit_code": result.returncode,
            "output": output,
            "summary": {"passed": passed, "failed": failed},
        }
    except Exception as e:
        return {"path": path, "success": False, "error": str(e)}


# ── Tool definitions for registry ──────────────────────────────────────

_SYSTEM_TOOLS = {
    "execute_shell": execute_shell,
    "read_file": read_file,
    "write_file": write_file,
    "edit_file": edit_file,
    "glob_files": glob_files,
    "list_directory": list_directory,
    "search_files": search_files,
    "execute_python": execute_python,
    "install_package": install_package,
    "run_tests": run_tests,
    "web_fetch": web_fetch,
    "web_search": web_search,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "list_processes": list_processes,
}

_SYSTEM_TOOL_DEFS = {
    "execute_shell": {
        "name": "execute_shell",
        "description": "Execute a shell command and return stdout, stderr, and exit code. Use this to run terminal commands, check system state, install system packages, manage files, etc. Dangerous commands are blocked.",
        "fn": execute_shell,
        "params_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "workdir": {"type": "string", "description": "Working directory for the command"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 60)"},
            },
            "required": ["command"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read file contents. Use this to inspect code, configuration, logs, or any text file. Returns content with line count and size info.",
        "fn": read_file,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative path to the file"},
                "lines": {"type": "integer", "description": "Max number of lines to return (useful for long files)"},
                "offset": {"type": "integer", "description": "Start reading from this line number", "default": 0},
            },
            "required": ["path"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content. Creates parent directories automatically. Use append=true to add content to end of file.",
        "fn": write_file,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Full content to write to the file"},
                "append": {"type": "boolean", "description": "Append to file instead of overwriting", "default": False},
            },
            "required": ["path", "content"],
        },
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Make a precise text replacement in a file. The old_string must match exactly (including whitespace/indentation). If the string appears multiple times, only the first occurrence is replaced. Use read_file first to verify the exact content to replace.",
        "fn": edit_file,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {"type": "string", "description": "The exact text to replace (must match precisely, including whitespace)"},
                "new_string": {"type": "string", "description": "The new text to replace it with"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    "glob_files": {
        "name": "glob_files",
        "description": "Find files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts', '*.md'). Returns relative file paths sorted alphabetically.",
        "fn": glob_files,
        "params_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match (e.g. '**/*.py', 'src/**/*.js')"},
                "path": {"type": "string", "description": "Base directory for the search", "default": "."},
            },
            "required": ["pattern"],
        },
    },
    "search_files": {
        "name": "search_files",
        "description": "Search for a regex pattern recursively in files (like grep -r). Returns matching file paths, line numbers, and the matching line content. Case-insensitive by default.",
        "fn": search_files,
        "params_schema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to search recursively"},
                "pattern": {"type": "string", "description": "Regex pattern to search for (case-insensitive)"},
                "file_glob": {"type": "string", "description": "File glob to filter (e.g. '*.py', '*.md')", "default": "*"},
                "max_results": {"type": "integer", "description": "Max results to return", "default": 50},
            },
            "required": ["directory", "pattern"],
        },
    },
    "execute_python": {
        "name": "execute_python",
        "description": "Execute Python code in a subprocess and return stdout, stderr, and exit code. Use this for calculations, data analysis, file processing, or running Python scripts. The code runs isolated.",
        "fn": execute_python,
        "params_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "workdir": {"type": "string", "description": "Working directory for execution"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30},
            },
            "required": ["code"],
        },
    },
    "install_package": {
        "name": "install_package",
        "description": "Install a Python package using pip. Use this to add dependencies needed for code execution or analysis.",
        "fn": install_package,
        "params_schema": {
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "Package name (e.g. 'requests', 'numpy')"},
                "upgrade": {"type": "boolean", "description": "Upgrade the package if already installed", "default": False},
            },
            "required": ["package"],
        },
    },
    "run_tests": {
        "name": "run_tests",
        "description": "Run tests in a directory. Supports pytest and unittest. Use this after making code changes to verify correctness.",
        "fn": run_tests,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the test directory or file", "default": "."},
                "test_type": {"type": "string", "enum": ["pytest", "unittest"], "description": "Test framework to use", "default": "pytest"},
                "extra_args": {"type": "string", "description": "Extra arguments passed to the test command"},
            },
        },
    },
    "web_fetch": {
        "name": "web_fetch",
        "description": "Fetch content from a URL and return as text. HTML is stripped for readability. Use this to read documentation, API responses, or any web page.",
        "fn": web_fetch,
        "params_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch"},
                "max_chars": {"type": "integer", "description": "Max characters to return", "default": 10000},
            },
            "required": ["url"],
        },
    },
    "web_search": {
        "name": "web_search",
        "description": "Search the web and return results with titles, URLs, and snippets. Use this to find current information, documentation, or answers to questions.",
        "fn": web_search,
        "params_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results to return", "default": 5},
            },
            "required": ["query"],
        },
    },
    "git_status": {
        "name": "git_status",
        "description": "Show git working tree status (modified, staged, untracked files).",
        "fn": git_status,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the git repository (default: current directory)"},
            },
        },
    },
    "git_diff": {
        "name": "git_diff",
        "description": "Show git diff of working tree changes. Set staged=true to see staged changes.",
        "fn": git_diff,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the git repository"},
                "staged": {"type": "boolean", "description": "Show staged changes instead of working tree", "default": False},
            },
        },
    },
    "git_log": {
        "name": "git_log",
        "description": "Show recent git commit log with hashes, dates, and messages.",
        "fn": git_log,
        "params_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the git repository"},
                "max_count": {"type": "integer", "description": "Max commits to show", "default": 10},
            },
        },
    },
    "list_processes": {
        "name": "list_processes",
        "description": "List running processes sorted by CPU or memory usage. Cross-platform (Windows/Linux/macOS).",
        "fn": list_processes,
        "params_schema": {
            "type": "object",
            "properties": {
                "sort_by": {"type": "string", "enum": ["cpu", "mem"], "description": "Sort by cpu or memory", "default": "cpu"},
                "limit": {"type": "integer", "description": "Max processes to show", "default": 20},
            },
        },
    },
}
