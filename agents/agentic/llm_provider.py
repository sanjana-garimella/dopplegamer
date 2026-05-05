"""LLM provider integration for agentic agents.

Supports:
- OpenAI (GPT-4, GPT-4o-mini, GPT-3.5)
- Ollama (local models)
- Mock (for testing without API keys)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import os


LLMProvider = Literal["openai", "ollama", "mock"]


@dataclass
class LLMConfig:
    provider: LLMProvider = "openai"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 500
    timeout: float = 30.0


class LLMBase(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate response from prompt."""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        ...


class OpenAILLM(LLMBase):
    """OpenAI GPT models via API."""
    
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI not installed. Install with: pip install openai"
            )
        
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key provided. Set OPENAI_API_KEY env var or pass api_key parameter."
            )
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate response using OpenAI API."""
        import openai
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )
            return response.choices[0].message.content or ""
        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in OpenAI generation: {e}") from e
    
    def close(self) -> None:
        pass


class OllamaLLM(LLMBase):
    """Local Ollama models."""
    
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        
        try:
            import requests
        except ImportError:
            raise ImportError("requests not installed. Install with: pip install requests")
        
        self.base_url = config.base_url or "http://localhost:11434"
        self.model = config.model
        
        # Verify connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(
                    f"Ollama not running. Start with: ollama serve\n"
                    f"Then pull a model: ollama pull {config.model}"
                )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Start with: ollama serve"
            )
    
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate response using Ollama."""
        import requests
        
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": self.config.temperature,
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama request error: {e}") from e
        except KeyError as e:
            raise RuntimeError(f"Invalid Ollama response: missing key {e}") from e
    
    def close(self) -> None:
        pass


class MockLLM(LLMBase):
    """Mock LLM for testing (no API calls)."""
    
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return deterministic mock response."""
        # Return action based on prompt keywords
        if "POWER" in prompt:
            return "ACTION: POWER"
        elif "RECHARGE" in prompt:
            return "ACTION: RECHARGE"
        elif "opponent" in prompt.lower() and "rock" in prompt.lower():
            return "ACTION: PAPER"
        elif "opponent" in prompt.lower() and "paper" in prompt.lower():
            return "ACTION: SCISSORS"
        elif "opponent" in prompt.lower() and "scissors" in prompt.lower():
            return "ACTION: ROCK"
        else:
            return "ACTION: ROCK"
    
    def close(self) -> None:
        pass


def get_llm(config: LLMConfig) -> LLMBase:
    """Factory function to get LLM provider."""
    if config.provider == "openai":
        return OpenAILLM(config)
    elif config.provider == "ollama":
        return OllamaLLM(config)
    elif config.provider == "mock":
        return MockLLM(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")


# Convenience functions
def generate_with_provider(
    prompt: str,
    provider: LLMProvider = "openai",
    model: str = "gpt-4o-mini",
    system: str | None = None,
    **kwargs
) -> str:
    """One-shot generation from any provider."""
    config = LLMConfig(provider=provider, model=model, **kwargs)
    llm = get_llm(config)
    try:
        return llm.generate(prompt, system)
    finally:
        llm.close()


# Usage example
if __name__ == "__main__":
    # Example 1: Mock (no API key needed)
    print("=== Mock LLM ===")
    mock_config = LLMConfig(provider="mock")
    mock_llm = get_llm(mock_config)
    response = mock_llm.generate("The opponent played rock. What should I play?")
    print(f"Response: {response}\n")
    
    # Example 2: OpenAI (requires OPENAI_API_KEY)
    try:
        print("=== OpenAI LLM ===")
        openai_config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7
        )
        openai_llm = get_llm(openai_config)
        response = openai_llm.generate(
            "The opponent has been playing paper frequently. What's the best counter?",
            system="You are an expert at rock-paper-scissors strategy."
        )
        print(f"Response: {response}\n")
        openai_llm.close()
    except (ImportError, ValueError) as e:
        print(f"OpenAI not configured: {e}\n")
    
    # Example 3: Ollama (requires ollama running locally)
    try:
        print("=== Ollama LLM ===")
        ollama_config = LLMConfig(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434"
        )
        ollama_llm = get_llm(ollama_config)
        response = ollama_llm.generate("What's a winning strategy for rock-paper-scissors?")
        print(f"Response: {response}\n")
        ollama_llm.close()
    except (ImportError, ConnectionError) as e:
        print(f"Ollama not available: {e}\n")
