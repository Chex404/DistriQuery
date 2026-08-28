"""Generation.

Defines an LLMClient interface plus two implementations:

- FakeLLMClient: deterministic, no API key or network call. Builds an
  "answer" by extracting the most relevant sentence from the top retrieved
  chunk. Used in tests and this demo.

- AnthropicLLMClient: the real thing. Requires:
      pip install anthropic
      export ANTHROPIC_API_KEY=...
"""

from abc import ABC, abstractmethod
from typing import List

from distriquery.vectorstore import SearchResult

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context doesn't contain the answer, say so — do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class FakeLLMClient(LLMClient):
    """Deterministic extractive stand-in — no network, no API key, no cost."""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "FakeLLMClient.generate() is not used directly — "
            "see pipeline.py's use of extractive_answer()."
        )

    @staticmethod
    def extractive_answer(question: str, results: List[SearchResult]) -> str:
        if not results:
            return "I couldn't find anything relevant to answer that question."

        best = results[0].chunk.text.strip()
        first_sentence = best.split(". ")[0].strip()
        return f"{first_sentence}. (extracted from the top-ranked retrieved chunk)"


class AnthropicLLMClient(LLMClient):
    """Real LLM calls via the Anthropic API. Requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # lazy import

        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def build_prompt(question: str, results: List[SearchResult]) -> str:
    context = "\n\n".join(
        f"[chunk {r.chunk.position} from {r.chunk.source}]\n{r.chunk.text}" for r in results
    )
    return PROMPT_TEMPLATE.format(context=context, question=question)


def get_llm_client(backend: str, model: str = "claude-sonnet-4-6") -> LLMClient:
    if backend == "fake":
        return FakeLLMClient()
    if backend == "anthropic":
        return AnthropicLLMClient(model=model)
    raise ValueError(f"Unknown LLM backend: {backend}")