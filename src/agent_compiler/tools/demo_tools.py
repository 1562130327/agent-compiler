"""Demonstration tools — simulated system operations for demo.

Each tool is a pure function: params in, dict out.
Register custom tools via ToolRegistry.register(name, fn).
"""

import random
from datetime import datetime, timedelta


def get_system_status(format: str = "summary") -> dict:
    return {
        "hostname": "demo-server",
        "cpu_percent": round(random.uniform(10, 60), 1),
        "memory_used_gb": round(random.uniform(4, 12), 1),
        "memory_total_gb": 16,
        "uptime_days": random.randint(1, 180),
        "active_services": ["nginx", "postgresql", "sshd"],
    }


def get_disk_usage(**kwargs) -> dict:
    return {
        "total_gb": 512,
        "used_gb": round(random.uniform(200, 450), 1),
        "free_gb": round(random.uniform(62, 312), 1),
        "use_percent": round(random.uniform(40, 88), 1),
        "mounts": [
            {"mount": "/", "used_pct": "45%"},
            {"mount": "/data", "used_pct": "78%"},
        ],
    }


def find_large_files(path: str = "/var/log", top_n: int = 10) -> dict:
    extensions = [".log", ".gz", ".tar", ".db", ".bak"]
    files = []
    for i in range(top_n):
        name = f"{path}/file_{i}{random.choice(extensions)}"
        size_mb = round(random.uniform(10, 2048), 2)
        mtime = datetime.now() - timedelta(days=random.randint(0, 90))
        files.append({"name": name, "size_mb": size_mb,
                       "modified": mtime.isoformat()})
    files.sort(key=lambda f: f["size_mb"], reverse=True)
    return {"path": path, "top_n": top_n, "files": files}


def search_logs(pattern: str = "ERROR", days: int = 1, level: str = "ERROR",
                max_lines: int = 200) -> dict:
    log_levels = ["ERROR", "WARNING", "CRITICAL", "FATAL"]
    entries = []
    for _ in range(random.randint(3, 15)):
        ts = datetime.now() - timedelta(hours=random.randint(0, days * 24))
        entries.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "level": random.choice(log_levels),
            "service": random.choice(["nginx", "postgresql", "app-server", "redis"]),
            "message": f"{pattern}: {random.choice([
                'Connection timeout after 30s',
                'Disk I/O error on device sda2',
                'Out of memory: process killed',
                'SSL certificate validation failed',
                'Rate limit exceeded for client',
                'Database connection pool exhausted',
                'Permission denied for user',
            ])}",
        })
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return {
        "pattern": pattern, "days": days, "level": level,
        "total_hits": len(entries), "entries": entries[:max_lines],
    }


def generate_report(format: str = "markdown", title: str = "报告",
                    include_timeline: bool = False) -> dict:
    if format == "markdown":
        body = f"# {title}\n\n"
        body += f"生成时间: {datetime.now().isoformat()}\n\n"
        body += "## 摘要\n\n- 系统运行正常\n- 检测到 3 条错误日志\n- 磁盘使用率 68%\n\n"
        if include_timeline:
            body += "## 时间线\n\n"
            body += "| 时间 | 事件 |\n|------|------|\n"
            for i, h in enumerate([24, 12, 6, 2, 1], 1):
                body += f"| {h}h 前 | 事件 #{i} |\n"
        body += "\n## 建议\n\n- 关注磁盘空间增长趋势\n- 检查 nginx 错误率\n"
    else:
        body = f"=== {title} ===\n生成时间: {datetime.now().isoformat()}\n"
    return {"format": format, "title": title, "body": body}


def list_directory(path: str = ".", limit: int = 50) -> dict:
    files = [
        {"name": "app.log", "size_kb": 1024, "modified": "2026-04-30"},
        {"name": "config.yaml", "size_kb": 2, "modified": "2026-04-28"},
        {"name": "data.db", "size_kb": 51200, "modified": "2026-05-01"},
        {"name": "error.log", "size_kb": 256, "modified": "2026-05-01"},
        {"name": "server.py", "size_kb": 8, "modified": "2026-04-15"},
    ]
    return {"path": path, "total": len(files), "files": files[:limit]}


def get_current_time(**kwargs) -> dict:
    now = datetime.now()
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


# Built-in tool registry for auto-registration
_BUILTIN_TOOLS = {
    "get_system_status": get_system_status,
    "get_disk_usage": get_disk_usage,
    "find_large_files": find_large_files,
    "search_logs": search_logs,
    "generate_report": generate_report,
    "list_directory": list_directory,
    "get_current_time": get_current_time,
}
