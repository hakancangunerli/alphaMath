"""
Calling LLM APIs

LLMs available:
- Groq: llama3-70b-8192, llama3-8b-8192, llama2-70b-4096, mixtral-8x7b-32768, gemma-7b-it
- OpenAI: gpt-3.5-turbo, gpt-4-turbo

Usage:
- call_llm_api(model, system_query, user_query, time_interval=1)
    Parameters:
    - model: the LLM to call
    - system_query: the system query
    - user_query: the user query
    - time_interval: do not call APIs twice within time_interval seconds (for openai and groq only. For ollama, it is not necessary to set this parameter.)
    
    Return:
    - the response (a string) from the LLM

TODO: add other parameters (e.g., temperature)
"""

import yaml
import openai
import ollama
from groq import Groq
from constants import LLMS, CODE_LLMS, LLMS_FROM_OPENAI
import time

# set-up api keys from the config file
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)
openai.api_key = cfg["api_key"]["openai"]
groq_client = Groq(api_key=cfg["api_key"]["groq"])


class LLMCLients:
    _instance = None

    def __new__(cls):
        # Singleton pattern, sets up api keys only once
        if cls._instance is None:
            with open("config.yaml") as f:
                cfg = yaml.safe_load(f)
            openai.api_key = cfg["api_key"]["openai"]
            cls.groq_client = Groq(api_key=cfg["api_key"]["groq"])
        return cls._instance


def call_llm_api(
    model: str,
    system_query: str = "",
    user_query: str = "",
    time_interval: float = 1,
):
    assert model in LLMS, f"LLM must be one of {LLMS}"

    # set up clients and api keys
    LLMCLients()

    if model in LLMS_FROM_OPENAI:
        t = time.time()
        res = (
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
        if time.time() - t < time_interval:
            time.sleep(time_interval - time.time() + t)
        return res

    elif model in CODE_LLMS:
        return ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_query},
                {"role": "user", "content": user_query},
            ],
        )["message"]["content"]

    else:
        t = time.time()
        res = (
            LLMCLients.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_query},
                    {"role": "user", "content": user_query},
                ],
            )
            .choices[0]
            .message.content
        )
        if time.time() - t < time_interval:
            time.sleep(time_interval - time.time() + t)
        return res
