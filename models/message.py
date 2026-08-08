from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Literal
import time
import uuid

Role = Literal["app/tts", "user"]

@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    role: Role = "app/tts"
    text: str = ""
    ts: float = field(default_factory=lambda: time.time())
    meta: Dict[str, Any] = field(default_factory=dict)