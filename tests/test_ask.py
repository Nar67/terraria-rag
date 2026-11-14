import pytest

from ask import ThinkFilter, cited_numbers, thinking_options

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
