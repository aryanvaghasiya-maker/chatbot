import re
from decouple import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage


class _ThinkStrippedLLM:
    """
    Thin wrapper around a ChatOpenAI instance that strips Qwen3-style
    <think>...</think> chain-of-thought blocks from every response before
    returning it to the caller, so downstream JSON parsers never see CoT tokens.
    """

    def __init__(self, llm: ChatOpenAI):
        self._llm = llm
        # Expose attributes that callers (e.g. ResumeAgent, tests) rely on
        self.model_name: str = llm.model_name

    @staticmethod
    def _strip_think(text: str) -> str:
        """Remove <think>…</think> blocks (including nested whitespace)."""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    def _clean_msg(self, msg: AIMessage) -> AIMessage:
        content = self._strip_think(msg.content) if isinstance(msg.content, str) else msg.content
        msg.content = content
        return msg

    def invoke(self, *args, **kwargs):
        return self._clean_msg(self._llm.invoke(*args, **kwargs))

    async def ainvoke(self, *args, **kwargs):
        msg = await self._llm.ainvoke(*args, **kwargs)
        return self._clean_msg(msg)

    def with_structured_output(self, *args, **kwargs):
        # Structured output goes through the underlying LLM directly
        return self._llm.with_structured_output(*args, **kwargs)

    # Delegate all other attribute access to the wrapped LLM
    def __getattr__(self, name):
        return getattr(self._llm, name)


def get_llm(temperature: float = 0.0, **kwargs) -> _ThinkStrippedLLM | ChatOpenAI:
    """
    Central factory function to instantiate ChatOpenAI compatible models.
    Supports either OpenAI or Groq (using OpenAI compatibility endpoint).

    For Groq models that emit <think> blocks (e.g. qwen/qwen3-32b), the
    returned instance automatically strips CoT tokens before returning.
    """
    groq_api_key = config("GROQ_API_KEY", default="").strip()
    openai_api_key = config("OPENAI_API_KEY", default="").strip()

    # Default to groq if GROQ_API_KEY is defined in env
    default_provider = "groq" if groq_api_key else "openai"
    provider = config("LLM_PROVIDER", default=default_provider).strip().lower()
    use_groq = config("USE_GROQ", cast=bool, default=False)

    if provider == "groq" or use_groq:
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is not defined or is empty.")
        groq_model = config("GROQ_MODEL", default="llama-3.3-70b-versatile").strip()

        args = {
            "model": groq_model,
            "api_key": groq_api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "temperature": temperature,
        }
        args.update(kwargs)
        base_llm = ChatOpenAI(**args)

        # Wrap with think-stripper for reasoning models (Qwen3, DeepSeek-R1, etc.)
        is_reasoning_model = any(name in groq_model.lower() for name in ["qwen3", "deepseek-r1", "r1-", "qwq"])
        return _ThinkStrippedLLM(base_llm) if is_reasoning_model else base_llm

    else:
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not defined or is empty.")
        openai_model = config("OPENAI_MODEL", default="gpt-3.5-turbo").strip()

        args = {
            "model": openai_model,
            "api_key": openai_api_key,
            "temperature": temperature,
        }
        args.update(kwargs)
        return ChatOpenAI(**args)
