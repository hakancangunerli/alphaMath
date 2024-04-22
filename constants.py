LLMS_FROM_GROQ = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama2-70b-4096",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]
LLMS_FROM_OPENAI = ["gpt-3.5-turbo", "gpt-4-turbo"]
LLMS = LLMS_FROM_GROQ + LLMS_FROM_OPENAI
DEFAULT_JUDGE_LLM = "gpt-3.5-turbo"
DEFAULT_SOLVER_LLM = "llama3-70b-8192"
