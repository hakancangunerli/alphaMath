LLMS_FROM_GROQ = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama2-70b-4096",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]
LLMS_FROM_OPENAI = ["gpt-3.5-turbo", "gpt-4-turbo"]
CODE_LLMS = ["codegemma"]
LLMS = LLMS_FROM_GROQ + LLMS_FROM_OPENAI + CODE_LLMS

# Making the default judge LLM to llama3-8b-8192 for experiments as gpt-3.5-turbo incur costs
# TODO: change back to gpt-3.5-turbo for production
DEFAULT_JUDGE_LLM = "llama3-8b-8192"
DEFAULT_SOLVER_LLM = "llama3-70b-8192"
ALL_PROBLEM_CLASSES = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
