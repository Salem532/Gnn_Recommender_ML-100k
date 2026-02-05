import yaml
from typing import Any, Dict

class Config:
    """
    配置类，用于解析 YAML 文件并提供属性访问方式。
    """
    def __init__(self, config_path: str = "", config_dict: Dict = {}):
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        elif config_dict:
            self._config = config_dict
        else:
            self._config = {}

    def __getattr__(self, name: str) -> Any:
        """
        允许通过 . 访问属性，例如 config.data
        """
        value = self._config.get(name)
        if isinstance(value, dict):
            return Config(config_dict=value)
        return value

    def to_dict(self) -> Dict:
        """返回原始字典"""
        return self._config

    def __repr__(self) -> str:
        return str(self._config)