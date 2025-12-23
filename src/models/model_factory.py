"""
AI Model Factory for Polymarket Trading Bot

Provides unified interface to multiple AI providers including:
- Anthropic Claude
- Google Gemini
- DeepSeek (via OpenRouter)
- Perplexity (sonar-pro with web search)

Author: Polymarket Trading Bot
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod
from functools import wraps

from anthropic import Anthropic
from openai import OpenAI
import httpx
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    genai = None


@dataclass
class ModelResponse:
    """Response from AI model with usage information."""
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    response_time: float = 0.0  # Response time in seconds
    cost_estimate: float = 0.0  # Estimated cost in USD
    success: bool = True
    error: Optional[str] = None


MODEL_COSTS = {
    # Claude (input, output per 1M tokens)
    "claude-3-5-haiku-20241022": (0.25, 1.25),
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    # Gemini (input, output per 1M tokens)
    "gemini-pro": (0.50, 1.50),
    "gemini-1.5-pro": (1.25, 5.0),
    # OpenAI (input, output per 1M tokens)
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-4o": (2.50, 10.0),
    "gpt-4-turbo": (10.0, 30.0),
    # DeepSeek via OpenRouter (input, output per 1M tokens)
    "deepseek/deepseek-v3.2": (0.24, 0.38),
    "deepseek/deepseek-chat": (0.14, 0.28),
    # Perplexity (input, output per 1M tokens)
    "sonar-pro": (3.0, 15.0),
    "sonar": (1.0, 1.0),
}


class BaseModel(ABC):
    """Abstract base class for AI model providers with efficiency features."""

    def __init__(self, model_name: str, api_key_env_var: str, cost_per_token: float = 0.0) -> None:
        """Initialize model with name and API key environment variable."""
        self.model_name = model_name
        self.api_key_env_var = api_key_env_var
        self.cost_per_token = cost_per_token

        # Load API key from environment
        self.api_key = os.getenv(api_key_env_var)
        if not self.api_key:
            logger.warning(f"Environment variable {api_key_env_var} not set - {model_name} will be skipped")
            self.api_key = None
        else:
            logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")

    def _retry_with_backoff(self, func, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs):
        """Execute function with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e

                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.1f}s: {e}")
                time.sleep(delay)

    def _estimate_cost(self, usage: Optional[Dict[str, Any]]) -> float:
        """Estimate cost based on token usage."""
        if not usage or not self.cost_per_token:
            return 0.0

        total_tokens = usage.get('total_tokens', 0)
        return total_tokens * self.cost_per_token

    def generate_response(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: int = 30
    ) -> ModelResponse:
        """Generate response from the AI model with efficiency features."""
        if not self.api_key:
            raise ValueError(f"API key not available for {self.model_name}")

        start_time = time.time()

        try:
            # Execute with retry logic and timeout
            result = self._retry_with_backoff(
                self._call_api,
                system_prompt,
                user_content,
                temperature,
                max_tokens,
                timeout=timeout
            )

            response_time = time.time() - start_time
            cost_estimate = self._estimate_cost(result.get('usage'))

            return ModelResponse(
                content=result['content'],
                model=self.model_name,
                usage=result.get('usage'),
                response_time=response_time,
                cost_estimate=cost_estimate,
                success=True,
                error=None
            )

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"{self.__class__.__name__} failed after {response_time:.1f}s: {e}")
            return ModelResponse(
                content="",
                model=self.model_name,
                usage=None,
                response_time=response_time,
                cost_estimate=0.0,
                success=False,
                error=str(e)
            )

    @abstractmethod
    def _call_api(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict[str, Any]:
        """Call the specific API - implemented by subclasses."""
        pass


class ClaudeModel(BaseModel):
    """Anthropic Claude model interface."""

    def __init__(self, model_name: str = "claude-3-5-haiku-20241022") -> None:
        # Claude pricing: $3/$15 per 1M tokens (input/output)
        super().__init__(model_name, "ANTHROPIC_API_KEY", cost_per_token=0.000009)
        if self.api_key:
            self.client = Anthropic(api_key=self.api_key)

    def _call_api(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict[str, Any]:
        """Call Claude API."""
        logger.debug(f"Calling Claude with model: {self.model_name}")

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            timeout=timeout
        )

        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens
        }

        return {
            "content": response.content[0].text,
            "usage": usage
        }


class GoogleModel(BaseModel):
    """Google Gemini 3 Pro model interface."""

    def __init__(self, model_name: str = "gemini-3-pro-preview") -> None:
        # Gemini 3 Pro pricing: $1.25/$5 per 1M tokens (input/output)
        super().__init__(model_name, "GOOGLE_API_KEY", cost_per_token=0.000003125)
        if self.api_key and GOOGLE_AVAILABLE:
            genai.configure(api_key=self.api_key)
        elif not GOOGLE_AVAILABLE:
            logger.warning("google-generativeai library not installed")

    def _call_api(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict[str, Any]:
        """Call Google Gemini API."""
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai library not installed")

        logger.debug(f"Calling Google Gemini with model: {self.model_name}")

        model = genai.GenerativeModel(self.model_name)

        # Combine system prompt and user content for Gemini
        full_prompt = f"System: {system_prompt}\n\nHuman: {user_content}"

        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
            request_options={"timeout": timeout}
        )

        # Estimate token usage (rough approximation)
        input_chars = len(full_prompt)
        output_chars = len(response.text)
        estimated_input_tokens = input_chars // 4  # Rough approximation
        estimated_output_tokens = output_chars // 4
        total_tokens = estimated_input_tokens + estimated_output_tokens

        usage = {
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": total_tokens,
            "input_chars": input_chars,
            "output_chars": output_chars
        }

        return {
            "content": response.text,
            "usage": usage
        }


class OpenAIModel(BaseModel):
    """OpenAI GPT-5.2 model interface."""

    def __init__(self, model_name: str = "gpt-5.2") -> None:
        # GPT-5.2 pricing: $5/$15 per 1M tokens (input/output)
        super().__init__(model_name, "OPENAI_API_KEY", cost_per_token=0.00001)
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def _call_api(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict[str, Any]:
        """Call OpenAI API directly."""
        logger.debug(f"Calling OpenAI with model: {self.model_name}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return {
            "content": choice.message.content,
            "usage": usage
        }


class OpenRouterModel(BaseModel):
    """Models via OpenRouter interface (DeepSeek, etc.)."""

    def __init__(self, model_name: str = "deepseek/deepseek-v3.2") -> None:
        # DeepSeek V3.2 pricing: $0.24/$0.38 per 1M tokens (input/output)
        super().__init__(model_name, "OPENROUTER_API_KEY", cost_per_token=0.00000031)
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )

    def _call_api(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict[str, Any]:
        """Call DeepSeek via OpenRouter API."""
        logger.debug(f"Calling DeepSeek via OpenRouter with model: {self.model_name}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return {
            "content": choice.message.content,
            "usage": usage
        }




class PerplexityModel(BaseModel):
    """Perplexity AI model interface with sonar-pro (web search built-in)."""

    def __init__(self, model_name: str = "sonar-pro") -> None:
        # Perplexity pricing: $3/$15 per 1M tokens (input/output)
        super().__init__(model_name, "PERPLEXITY_API_KEY", cost_per_token=0.000009)
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.perplexity.ai"
            )

    def _call_api(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float,
        max_tokens: int,
        timeout: int
    ) -> Dict[str, Any]:
        """Call Perplexity API with sonar-pro model."""
        logger.debug(f"Calling Perplexity with model: {self.model_name}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )

        choice = response.choices[0]
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }

        return {
            "content": choice.message.content,
            "usage": usage
        }


class ModelFactory:
    """Factory for creating AI model instances."""

    # Model type mappings
    MODEL_CLASSES = {
        "claude": ClaudeModel,
        "google": GoogleModel,       # For gemini models
        "openai": OpenAIModel,       # For gpt models
        "openrouter": OpenRouterModel, # For models via OpenRouter (DeepSeek, etc.)
        "perplexity": PerplexityModel,
    }

    # Default models for each provider
    DEFAULT_MODELS = {
        "claude": "claude-3-5-haiku-20241022",
        "google": "gemini-pro",
        "openai": "gpt-3.5-turbo",
        "openrouter": "deepseek/deepseek-v3.2",  # OpenRouter uses deepseek
        "deepseek": "deepseek/deepseek-v3.2",
        "perplexity": "sonar-pro"
    }

    def get_model(self, model_type: str, model_name: Optional[str] = None) -> BaseModel:
        """
        Get a model instance by type and name.

        Args:
            model_type: Type of model (claude, google, deepseek, perplexity)
            model_name: Specific model name (optional, uses default if not provided)

        Returns:
            Model instance

        Raises:
            ValueError: If model_type is not supported
        """
        model_type = model_type.lower()

        if model_type not in self.MODEL_CLASSES:
            supported = ", ".join(self.MODEL_CLASSES.keys())
            raise ValueError(f"Unsupported model type: {model_type}. Supported: {supported}")

        # Use default model if not specified
        if model_name is None:
            model_name = self.DEFAULT_MODELS[model_type]

        logger.info(f"Creating {model_type} model: {model_name}")

        model_class = self.MODEL_CLASSES[model_type]
        return model_class(model_name)

    def check_available_models(self) -> Dict[str, bool]:
        """
        Check which models are available based on API key presence.

        Returns:
            Dict mapping model types to availability status
        """
        return {
            "claude": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
        }

    def test_models(self, quick_test: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Test all available models with a simple prompt.

        Args:
            quick_test: If True, use a very short prompt for faster testing

        Returns:
            Dict with test results for each model
        """
        test_prompt = "Say 'Hello from AI!' in exactly 3 words."
        system_prompt = "You are a helpful AI assistant."
        temperature = 0.1
        max_tokens = 50 if quick_test else 2048

        results = {}
        available_models = self.check_available_models()

        logger.info("🧪 Testing AI models...")

        for model_type, is_available in available_models.items():
            if not is_available:
                results[model_type] = {
                    "success": False,
                    "error": "API key not available",
                    "response_time": 0.0,
                    "cost_estimate": 0.0
                }
                continue

            try:
                logger.info(f"Testing {model_type} model...")
                model = self.get_model(model_type)

                response = model.generate_response(
                    system_prompt=system_prompt,
                    user_content=test_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=15
                )

                results[model_type] = {
                    "success": True,
                    "content": response.content[:100] + "..." if len(response.content) > 100 else response.content,
                    "response_time": response.response_time,
                    "cost_estimate": response.cost_estimate,
                    "model": response.model,
                    "usage": response.usage
                }

                logger.info(f"✅ {model_type} test successful ({response.response_time:.1f}s, ${response.cost_estimate:.4f})")

            except Exception as e:
                results[model_type] = {
                    "success": False,
                    "error": str(e),
                    "response_time": 0.0,
                    "cost_estimate": 0.0
                }
                logger.error(f"❌ {model_type} test failed: {e}")

        # Summary
        successful = sum(1 for r in results.values() if r["success"])
        total_cost = sum(r.get("cost_estimate", 0) for r in results.values())
        total_time = sum(r.get("response_time", 0) for r in results.values())

        logger.info(f"🧪 Model testing complete: {successful}/{len(results)} successful")
        logger.info(f"💰 Total cost: ${total_cost:.4f} | Total time: {total_time:.1f}s")

        return results


def run_self_tests():
    """Run self-tests for model factory functions."""
    print("🧪 Model Factory Self-Tests")

    factory = ModelFactory()

    # Test that all expected providers are supported
    expected_providers = ['claude', 'google', 'openai', 'openrouter', 'perplexity']
    for provider in expected_providers:
        assert provider in factory.MODEL_CLASSES, f"Provider {provider} should be supported"

    # Test that default models are defined
    for provider in expected_providers:
        assert provider in factory.DEFAULT_MODELS, f"Default model for {provider} should be defined"

    # Test that factory has expected methods
    assert hasattr(factory, 'get_model'), "Factory should have get_model method"
    assert callable(getattr(factory, 'get_model')), "get_model should be callable"

    print("✅ Model Factory self-tests passed")


# Singleton instance
model_factory = ModelFactory()


if __name__ == "__main__":
    run_self_tests()
