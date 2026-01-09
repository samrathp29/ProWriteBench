"""
Abstract base class for LLM models in ProWriteBench.

All model adapters must inherit from BaseModel and implement the generate method.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseModel(ABC):
    """Abstract base class for LLM models."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the model.

        Args:
            model_name: Name/identifier of the model (e.g., "claude-opus-4-5", "gpt-4")
            api_key: API key for the model provider (if None, will try to load from environment)
            **kwargs: Additional model-specific configuration
        """
        self.model_name = model_name
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text from the model.

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional generation parameters

        Returns:
            Generated text as a string

        Raises:
            Exception: If generation fails
        """
        pass

    def generate_with_metadata(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text and return with metadata (token usage, timing, etc.).

        Args:
            prompt: Input prompt for the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0-1.0)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary containing:
                - text: Generated text
                - metadata: Model-specific metadata (tokens, timing, etc.)
        """
        import time
        start_time = time.time()

        text = self.generate(prompt, max_tokens, temperature, **kwargs)

        end_time = time.time()

        return {
            "text": text,
            "metadata": {
                "model": self.model_name,
                "generation_time": end_time - start_time,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
