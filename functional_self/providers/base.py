"""Provider-independent model interface. The architecture must not depend on one model."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class BrainProvider(ABC):
    name: str = "base"
    calls: int = 0

    @abstractmethod
    def complete(self, task: str, system: str, user: str, context: Optional[dict] = None) -> str: ...

    def complete_json(self, task: str, system: str, user: str, model: Type[T],
                      context: Optional[dict] = None, retries: int = 2) -> T:
        """Ask for JSON matching `model`. Application code validates; invalid output is retried
        then raised — model prose is never parsed heuristically."""
        schema = json.dumps(model.model_json_schema(), indent=0)
        sys_prompt = (f"{system}\n\nRespond with ONLY a JSON object matching this schema "
                      f"(no prose, no code fences):\n{schema}")
        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.complete(task, sys_prompt, user, context)
            try:
                return model.model_validate(_extract_json(raw))
            except (ValidationError, ValueError) as e:
                last_err = e
                user = user + f"\n\nYour previous output was invalid ({str(e)[:200]}). Output valid JSON only."
        raise RuntimeError(f"provider {self.name} failed to produce valid {model.__name__}: {last_err}")


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("no JSON object found")
        return json.loads(m.group(0))
