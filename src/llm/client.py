"""
LLM Client -- Unified wrapper for NVIDIA, Groq, Gemini, OpenAI, and Ollama.

All agent reasoning goes through this module, which injects
the agent's personality (system prompt) and handles the LLM call.

Supported providers:
- nvidia    : NVIDIA build.nvidia.com (Nemotron, Llama, etc.) -- FREE credits!
- groq      : Groq Cloud (Llama 3.1, Gemma) -- FREE tier
- gemini    : Google AI Studio (Gemini 2.0 Flash) -- FREE tier (15 RPM)
- openai    : OpenAI API (GPT-4o-mini, etc.) -- paid
- ollama    : Local models via Ollama (unlimited, free)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# Provider configurations -- base URLs and default models
PROVIDER_CONFIGS = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "nvidia/nemotron-3-ultra-550b-a55b",
        "api_key_env": "NVIDIA_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "gemma2-9b-it",
        "api_key_env": "GROQ_API_KEY",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "default_model": "gemini-2.0-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "openai": {
        "base_url": None,  # uses default
        "default_model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
}


class LLMClient:
    """
    Unified LLM client supporting NVIDIA, Groq, Gemini, OpenAI, and Ollama.

    Usage:
        # Using NVIDIA Nemotron (free credits)
        client = LLMClient(provider="nvidia")

        # Using Groq with Gemma (free tier)
        client = LLMClient(provider="groq")

        # Using Gemini (free tier -- replaces OpenAI)
        client = LLMClient(provider="gemini")

        # Using local Ollama
        client = LLMClient(provider="ollama")

        response = client.chat("You are a scientist.", "Solve this problem...")
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "nvidia")
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider in PROVIDER_CONFIGS:
            config = PROVIDER_CONFIGS[self.provider]
            self.model = model or os.getenv(
                f"{self.provider.upper()}_MODEL", config["default_model"]
            )
            self._init_openai_compatible(
                base_url=config["base_url"],
                api_key_env=config["api_key_env"],
            )
        elif self.provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL", "llama3.1")
            self._init_ollama()
        else:
            raise ValueError(
                f"Unknown LLM provider: {self.provider}. "
                f"Supported: {', '.join(list(PROVIDER_CONFIGS.keys()) + ['ollama'])}"
            )

        # Track cumulative token usage
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def _init_openai_compatible(self, base_url: str | None, api_key_env: str) -> None:
        """Initialize an OpenAI-compatible client (works for NVIDIA, Groq, Gemini, OpenAI)."""
        try:
            from openai import OpenAI
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(
                    f"{api_key_env} not found. Set it in .env or environment.\n"
                    f"Get a free key:\n"
                    f"  NVIDIA:  https://build.nvidia.com\n"
                    f"  Groq:    https://console.groq.com\n"
                    f"  Gemini:  https://aistudio.google.com/apikey\n"
                    f"  OpenAI:  https://platform.openai.com"
                )

            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url

            self._client = OpenAI(**kwargs)
            logger.info(f"Initialized {self.provider} client (model: {self.model})")
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _init_ollama(self) -> None:
        """Initialize Ollama client."""
        try:
            import ollama
            self._ollama = ollama
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            self._client = ollama.Client(host=host)
        except ImportError:
            raise ImportError("Install ollama: pip install ollama")

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            system_prompt: The agent's personality/instruction prompt.
            user_message: The task or message to respond to.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            LLMResponse with the generated content.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        if self.provider in PROVIDER_CONFIGS:
            return self._chat_openai_compatible(messages, temp, tokens)
        else:
            return self._chat_ollama(messages, temp, tokens)

    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    import httpx

    @retry(
        wait=wait_exponential(multiplier=2, min=5, max=60),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _chat_openai_compatible(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> LLMResponse:
        """Chat completion via OpenAI-compatible API with auto-retries."""
        try:
            # Build kwargs -- NVIDIA Nemotron supports streaming with thinking
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # NVIDIA Nemotron models support reasoning/thinking mode and require specific parameters
            if self.provider == "nvidia" and "nemotron" in self.model.lower():
                # Nemotron fails with 500 if we don't provide the correct extra_body
                kwargs["temperature"] = min(temperature, 0.95)  # Nemotron caps at ~0.95
                kwargs["top_p"] = 0.95
                kwargs["max_tokens"] = max(max_tokens, 16384) # Ensure enough tokens for thinking
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": True},
                    "reasoning_budget": min(kwargs["max_tokens"], 16384)
                }

            response = self._client.chat.completions.create(**kwargs)

            usage = response.usage
            prompt_tok = usage.prompt_tokens if usage else 0
            completion_tok = usage.completion_tokens if usage else 0

            self.total_prompt_tokens += prompt_tok
            self.total_completion_tokens += completion_tok
            self.total_calls += 1

            return LLMResponse(
                content=response.choices[0].message.content or "",
                model=self.model,
                prompt_tokens=prompt_tok,
                completion_tokens=completion_tok,
                total_tokens=prompt_tok + completion_tok,
            )
        except Exception as e:
            logger.error(f"{self.provider} API error: {e}")
            raise

    def _chat_ollama(
        self, messages: list[dict], temperature: float, max_tokens: int
    ) -> LLMResponse:
        """Ollama chat completion."""
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_predict": max_tokens},
            )

            self.total_calls += 1

            return LLMResponse(
                content=response["message"]["content"],
                model=self.model,
            )
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise

    def get_usage_stats(self) -> dict:
        """Get cumulative usage statistics."""
        return {
            "provider": self.provider,
            "model": self.model,
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
