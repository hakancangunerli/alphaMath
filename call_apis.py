"""
Calling LLM APIs 

LLMs available:
- Groq: llama3-70b-8192, llama3-8b-8192, llama2-70b-4096, mixtral-8x7b-32768, gemma-7b-it
- OpenAI: gpt-3.5-turbo, gpt-4-turbo

Usage:
- call_llm_api(model, system_query, user_query)
    Parameters:
    - model: the LLM to call
    - system_query: the system query
    - user_query: the user query
    Return:
    - the response (a string) from the LLM

TODO: add other parameters (e.g., temperature)
"""

import yaml
import openai
from groq import Groq

LLMS_FROM_GROQ = [
    "llama3-70b-8192",
    "llama3-8b-8192",
    "llama2-70b-4096",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]
LLMS_FROM_OPENAI = ["gpt-3.5-turbo", "gpt-4-turbo"]
LLMS = LLMS_FROM_GROQ + LLMS_FROM_OPENAI

# set-up api keys from the config file
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
openai.api_key = cfg["api_key"]["openai"]
groq_client = Groq(api_key=cfg["api_key"]["groq"])


def call_llm_api(
    model: str,
    system_query: str = "",
    user_query: str = "",
):
    assert model in LLMS, f"LLM must be one of {LLMS}"
    if model in LLMS_FROM_OPENAI:
        return (
            openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_query},
                    {"role": "user", "content": user_query},
                ],
            )
            .choices[0]
            .message["content"]
        )
    else:
        return (
            groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_query},
                    {"role": "user", "content": user_query},
                ],
            )
            .choices[0]
            .message.content
        )

if __name__ == "__main__":
    print(
        call_llm_api(
            model='llama3-70b-8192',
            system_query="Hi! I am a student who is trying to solve a math problem. Can you help me?",
            user_query="Let $\\mathbf{a} = \\begin{pmatrix} -3 \\\\ 10 \\\\ 1 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} 5 \\\\ \\pi \\\\ 0 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} -2 \\\\ -2 \\\\ 7 \\end{pmatrix}.$  Compute\n\\[(\\mathbf{a} - \\mathbf{b}) \\cdot [(\\mathbf{b} - \\mathbf{c}) \\times (\\mathbf{c} - \\mathbf{a})].\\]",
        )
    )
