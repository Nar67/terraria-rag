# C: most literal: greedy decoding (temperature 0) and no presence penalty.
FROM qwen3.5:9b
PARAMETER temperature 0
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER presence_penalty 0
