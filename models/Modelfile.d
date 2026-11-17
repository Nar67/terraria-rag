# D: like A but without the presence penalty (isolates its effect).
FROM qwen3.5:9b
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER presence_penalty 0
# Safety cap on output length (DEVLOG §20): a request without its own limit
# can never generate for minutes.
PARAMETER num_predict 4096
