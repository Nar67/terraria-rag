# B: Qwen3.5-9B model card, non-thinking mode, general tasks.
FROM qwen3.5:9b
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER min_p 0
PARAMETER presence_penalty 1.5
# Safety cap on output length (DEVLOG §20): a request without its own limit
# can never generate for minutes.
PARAMETER num_predict 4096
