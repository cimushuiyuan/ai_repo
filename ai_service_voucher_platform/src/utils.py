import yaml
from typing import Any, Dict

def load_yaml(filepath: str) -> Dict[str, Any]:
    """安全地加载YAML文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)