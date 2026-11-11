import math
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List
from .base_tool import LocalTool

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


def calculate(expression: str) -> float:
    """Safely evaluates a basic mathematical expression.
    
    Args:
        expression: A string mathematical expression like '2 * 3.14 * 10' or 'sqrt(144) + 12'.
    """
    allowed_names = {
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
        "pi": math.pi, "e": math.e, "pow": math.pow, "abs": abs,
    }
    # Validate characters
    clean_expr = expression.strip()
    return float(eval(clean_expr, {"__builtins__": {}}, allowed_names))


def inspect_gpu_and_hardware() -> Dict[str, Any]:
    """Retrieves current NVIDIA GPU VRAM usage and system memory telemetry."""
    info: Dict[str, Any] = {
        "gpu_detected": False,
        "gpu_name": "Unknown",
        "total_vram_mb": 0,
        "used_vram_mb": 0,
        "free_vram_mb": 0,
    }
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info["gpu_detected"] = True
            info["gpu_name"] = name if isinstance(name, str) else name.decode()
            info["total_vram_mb"] = round(mem.total / (1024 * 1024), 1)
            info["used_vram_mb"] = round(mem.used / (1024 * 1024), 1)
            info["free_vram_mb"] = round(mem.free / (1024 * 1024), 1)
            pynvml.nvmlShutdown()
        except Exception as e:
            info["gpu_error"] = str(e)
            
    # Add disk and RAM info
    disk = shutil.disk_usage(".")
    info["free_disk_gb"] = round(disk.free / (1024**3), 2)
    return info


def inspect_file(file_path: str, max_lines: int = 100) -> str:
    """Reads the first N lines of a file safely.
    
    Args:
        file_path: Relative or absolute path to the file.
        max_lines: Max number of lines to return.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        return f"Error: File '{file_path}' does not exist."
    if not path.is_file():
        return f"Error: Path '{file_path}' is not a regular file."
    
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(max_lines)]
        return "".join(lines)
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


from .web_tools import web_search, fetch_web_page

# Export registered LocalTool instances and callables for Google ADK
LOCAL_TOOL_FUNCTIONS = [
    calculate,
    inspect_gpu_and_hardware,
    inspect_file,
    web_search,
    fetch_web_page,
]

LOCAL_TOOLS = [LocalTool(f) for f in LOCAL_TOOL_FUNCTIONS]
