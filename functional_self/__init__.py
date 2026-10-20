"""functional-self: persistent, evidence-grounded self-modeling for LLM agents."""
from .agent import FunctionalSelfAgent
from .config import Config
from .providers.mock_provider import MockProvider

__all__ = ["FunctionalSelfAgent", "Config", "MockProvider"]
__version__ = "0.1.0"
