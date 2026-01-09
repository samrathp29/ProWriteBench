"""
OpenAI model adapter for ProWriteBench.
"""

import os
from typing import Optional, Dict, Any
from openai import OpenAI

from .base import BaseModel


class OpenAIModel(BaseModel):
    """Adapter for OpenAI models (GPT-4, GPT-5, etc.)."""

    def __init__(
        self,
        model_name: str = "gpt-5",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize OpenAI model.

        Args:
            model_name: Name of OpenAI model (e.g., "gpt-4", "gpt-5", "gpt-4-turbo")
            api_key: OpenAI API key (if None, loads from OPENAI_API_KEY env var)
            **kwargs: Additional configuration
        """
        super().__init__(model_name, api_key, **kwargs)

        # Get API key from environment if not provided
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if self.api_key is None:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters (system, stop, etc.)

        Returns:
            Generated text

        Raises:
            Exception: If API call fails
        """
        try:
            # Extract system message if provided
            system = kwargs.pop("system", None)

            # Build messages
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # Create completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Extract text from response
            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

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

            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            end_time = time.time()

            return {
                "text": response.choices[0].message.content,
                "metadata": {
                    "model": self.model_name,
                    "generation_time": end_time - start_time,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                }
            }

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
