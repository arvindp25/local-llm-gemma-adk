import inspect
from typing import Any, Callable, Dict, Optional, get_type_hints
from pydantic import BaseModel
from core.exceptions import ToolExecutionError


def python_type_to_json_type(py_type: Any) -> str:
    """Map python types to OpenAPI / JSON Schema types."""
    if py_type in (int,):
        return "integer"
    elif py_type in (float,):
        return "number"
    elif py_type in (str,):
        return "string"
    elif py_type in (bool,):
        return "boolean"
    elif py_type in (list, List):
        return "array"
    elif py_type in (dict, Dict):
        return "object"
    return "string"


class LocalTool:
    """Wrapper around a Python callable to expose OpenAPI function schemas."""

    def __init__(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        self.schema = self._generate_schema()

    def _generate_schema(self) -> Dict[str, Any]:
        sig = inspect.signature(self.func)
        try:
            hints = get_type_hints(self.func)
        except Exception:
            hints = {}

        properties: Dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            
            p_type = hints.get(param_name, str)
            json_type = python_type_to_json_type(p_type)
            
            prop_def: Dict[str, Any] = {"type": json_type}
            properties[param_name] = prop_def

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def execute(self, **kwargs) -> Any:
        try:
            return self.func(**kwargs)
        except Exception as e:
            raise ToolExecutionError(f"Error executing tool '{self.name}': {str(e)}") from e
