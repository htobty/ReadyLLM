"""推理引擎统一适配层"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineInfo:
    name: str
    installed: bool
    version: str = ""


@dataclass
class StartParams:
    model_path: str
    extra_args: list[str] = field(default_factory=list)
    port: int = 8989
    host: str = "0.0.0.0"


class EngineAdapter(ABC):
    """推理引擎统一接口"""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check_installed(self) -> bool: ...

    @abstractmethod
    def start(self, params: StartParams) -> tuple[bool, str]: ...

    @abstractmethod
    def stop(self) -> tuple[bool, str]: ...

    @abstractmethod
    def is_running(self) -> bool: ...

    @abstractmethod
    def get_metrics_url(self) -> str: ...
