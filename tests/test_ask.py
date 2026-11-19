from types import SimpleNamespace

import pytest

import ask

from ask import MAX_ANSWER_TOKENS, MAX_THINKING_TOKENS, ThinkFilter, cited_numbers, max_answer_tokens, thinking_options

STREAM = "<think>plan: cite [2]</think>Answer [2] uses a < sign and <b>tag</b>."


@pytest.mark.parametrize("piece_size", [1, 2, 3, 7, 100])
def test_think_filter_handles_tags_split_across_pieces(piece_size):
    """Streamed text arrives in arbitrary pieces; a tag split over two pieces
    once made the whole answer disappear into the reasoning."""
    reasoning, answer = [], []
    f = ThinkFilter(reasoning.append, answer.append)
    for i in range(0, len(STREAM), piece_size):
        f.feed(STREAM[i:i + piece_size])
    f.flush()
    assert "".join(reasoning) == "plan: cite [2]"
    assert "".join(answer) == "Answer [2] uses a < sign and <b>tag</b>."


@pytest.mark.parametrize("answer, count, cited", [
    ("Use a Suspicious Looking Eye [2][6] at night [2].", 8, [2, 6]),
    ("Out of range [9] and [0] are ignored [3].", 8, [3]),
    ("No citations at all.", 8, []),
])
def test_cited_numbers(answer, count, cited):
    assert cited_numbers(answer, count) == cited


@pytest.mark.parametrize("think, options", [
    (None, {}),
    (True, {"chat_template_kwargs": {"enable_thinking": True}}),
    # Ollama only honours reasoning_effort, so --no-think sends both switches
    (False, {"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"}),
])
def test_thinking_options(think, options):
    assert thinking_options(think) == options


@pytest.mark.parametrize("think, cap", [
    (False, MAX_ANSWER_TOKENS),
    (True, MAX_THINKING_TOKENS),
    (None, MAX_THINKING_TOKENS),  # server default: Qwen3.5 reasons, so leave room for it
])
def test_every_answer_is_capped(think, cap):
    """A request without a cap once generated 29k tokens and froze the PC."""
    assert max_answer_tokens(think) == cap <= 4096


class FakeStream:
    """Stands in for the OpenAI client's streamed chat completion."""
    def __init__(self, reasoning, answer, finish=None):
        self.pieces = [("reasoning", t) for t in reasoning] + [("content", t) for t in answer]
        self.finish, self.closed = finish, False

    def __iter__(self):
        for n, (kind, text) in enumerate(self.pieces):
            if self.closed:
                return
            last = n == len(self.pieces) - 1
            delta = SimpleNamespace(content=text if kind == "content" else None,
                                    reasoning=text if kind == "reasoning" else None)
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=self.finish if last else None)])

    def close(self):
        self.closed = True


def fake_client(stream):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: stream)))


@pytest.mark.parametrize("reasoning_pieces, answer, think, forced", [
    (10, ["Use a Suspicious Looking Eye [2]."], True, False),   # short reasoning: untouched
    (ask.REASONING_BUDGET + 50, [], True, True),                # looping reasoning: stopped and forced
    (ask.REASONING_BUDGET + 50, [], False, False),              # reasoning off: no budget applies
])
def test_reasoning_budget(monkeypatch, reasoning_pieces, answer, think, forced):
    """A model that reasons past the budget gets stopped and made to answer
    (8 of 92 answers were empty without this)."""
    calls = []

    def fake_finish(model, messages, reasoning, on_answer, temperature=None, seed=None):
        calls.append(reasoning)
        on_answer("Forced answer [1].")
        return False

    monkeypatch.setattr(ask, "finish_after_reasoning", fake_finish)
    stream = FakeStream(["x"] * reasoning_pieces, answer)
    result = ask.generate(fake_client(stream), "m", [], think)
    assert result["budget_forced"] is forced
    assert stream.closed is forced
    assert result["answer"] == ("Forced answer [1]." if forced else "".join(answer))
    if forced:
        assert len(calls[0]) == ask.REASONING_BUDGET  # continues from exactly the reasoning so far


def test_chatml_matches_qwen_format():
    messages = [{"role": "system", "content": "Rules."}, {"role": "user", "content": "Q?"}]
    assert ask.chatml(messages) == "<|im_start|>system\nRules.<|im_end|>\n<|im_start|>user\nQ?<|im_end|>\n"


@pytest.mark.parametrize("env, think, model", [
    (None, None, ask.THINK_MODEL),    # server default reasons: sampling model
    (None, True, ask.THINK_MODEL),
    (None, False, ask.ANSWER_MODEL),  # no reasoning: deterministic model
    ("my-model", True, "my-model"),   # RAG_LLM_MODEL always wins
])
def test_model_for(monkeypatch, env, think, model):
    """Reasoning at temperature 0 looped on up to a third of the questions."""
    monkeypatch.setattr(ask, "LLM_MODEL", env)
    assert ask.model_for(think) == model
