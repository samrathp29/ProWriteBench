"""
Anthropic Claude model adapter for ProWriteBench.
"""

import os
from typing import Optional, Dict, Any
from anthropic import Anthropic

from .base import BaseModel


class AnthropicModel(BaseModel):
    """Adapter for Anthropic Claude models."""

    def __init__(
        self,
        model_name: str = "claude-opus-4-5",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Anthropic model.

        Args:
            model_name: Name of Claude model (e.g., "claude-opus-4-5", "claude-sonnet-4-5")
            api_key: Anthropic API key (if None, loads from ANTHROPIC_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)

        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        # Initialize Anthropic client
        self.client = Anthropic(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using Claude model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (system, stop_sequences, etc.)

        Returns:
            Generated text

        Raises:
            Exception: If API call fails
        """
        try:
            # Extract system message if provided
            system = kwargs.pop("system", None)

            # Create message
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **({"system": system} if system else {}),
                **kwargs
            )

            # Extract text from response
            return message.content[0].text

        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

    def generate_with_metadata(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with detailed metadata.

        Returns:
            Dictionary with text and metadata including token usage
        """
        import time
        start_time = time.time()

        try:
            system = kwargs.pop("system", None)

            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **({"system": system} if system else {}),
                **kwargs
            )

            end_time = time.time()

            return {
                "text": message.content[0].text,
                "metadata": {
                    "model": self.model_name,
                    "generation_time": end_time - start_time,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens,
                    "stop_reason": message.stop_reason,
                }
            }

        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
