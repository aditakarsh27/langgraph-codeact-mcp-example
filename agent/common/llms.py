from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSerializable
from openai import (
    APIStatusError as OpenAIApiStatusError,
    RateLimitError as OpenAIRateLimitError,
    APITimeoutError as OpenAIApiTimeoutError,
    APIConnectionError as OpenAIApiConnectionError,
    InternalServerError as OpenAIInternalServerError
)
from anthropic import (
    APIStatusError as AnthropicAPIStatusError,
    RateLimitError as AnthropicRateLimitError,
    APITimeoutError as AnthropicAPITimeoutError,
    APIConnectionError as AnthropicAPIConnectionError,
    InternalServerError as AnthropicInternalServerError
)
from agent.common.config import LLM_PROVIDER, REFLECTION_LLM_PROVIDER

# To solve the `Overloaded` error
def with_anthropic_retry(model):
    """Wraps an Anthropic model with retry logic for common API errors"""
    return model.with_retry(
        retry_if_exception_type=(
            AnthropicAPIStatusError,
            AnthropicRateLimitError,
            AnthropicAPITimeoutError,
            AnthropicAPIConnectionError,
            AnthropicInternalServerError
        ),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

# To solve the `Overloaded` error
def with_openai_retry(model):
    """Wraps an OpenAI model with retry logic for common API errors"""
    return model.with_retry(
        retry_if_exception_type=(
            OpenAIApiStatusError,
            OpenAIRateLimitError,
            OpenAIApiTimeoutError,
            OpenAIApiConnectionError,
            OpenAIInternalServerError
        ),
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

def get_react_agent_model() -> RunnableSerializable:
    """Returns an LLM optimized for extracting topics from content"""
    if LLM_PROVIDER == "openai":
        model = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.0,
            max_tokens=8192,
            disable_streaming=True,
        )
    elif LLM_PROVIDER == "anthropic":
        model = ChatAnthropic(
            model="claude-4-sonnet-20250514",
            temperature=0.0,
            max_tokens=8192,
            disable_streaming=True,
        )
    else:
        raise ValueError(f"Invalid LLM provider: {LLM_PROVIDER}")
    # TODO: Sort out the retry logic - currently doesn't allow due to RunnableSerializable.bind_tools not existing
    # return with_anthropic_retry(model)
    return model

def get_reflection_model() -> RunnableSerializable:
    """Returns an LLM optimized for extracting topics from content"""
    if REFLECTION_LLM_PROVIDER == "openai":
        model = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.0,
            max_tokens=8192,
            disable_streaming=True,
        )
    elif REFLECTION_LLM_PROVIDER == "anthropic":
        model = ChatAnthropic(
            model="claude-4-sonnet-20250514",
            temperature=0.0,
            max_tokens=8192,
            disable_streaming=True,
        )
    else:
        raise ValueError(f"Invalid LLM provider: {REFLECTION_LLM_PROVIDER}")
    return with_anthropic_retry(model)
