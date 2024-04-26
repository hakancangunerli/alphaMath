import io
import re
import sys
from call_apis import *
from logging_config import setup_logging
import logging


def remove_python_header(code_output):
    """Remove any line that only contains 'python'"""
    code_output = re.sub(
        r"^python\s*$", "", code_output, flags=re.IGNORECASE | re.MULTILINE
    )
    return code_output.strip()


def extract_code_blocks(text):
    # Regex pattern to find text enclosed within triple backticks
    pattern = r"```(.*?)```"

    # re.DOTALL allows dot (.) to match newline characters as well
    code_blocks = re.findall(pattern, text, re.DOTALL)

    return code_blocks[0] if code_blocks else text


def exec_capture_stdout(code):
    """
    Function to execute code and capture stdout
    Input: code
    Output:
        (True, stdout) if the code executes successfully,
        (False, error message) if otherwise
    """
    captured_output = io.StringIO()

    # Save the current stdout so we can restore it later
    original_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        exec(code)
        sys.stdout = original_stdout
        return True, captured_output.getvalue()
    except Exception as e:
        sys.stdout = original_stdout
        return False, e


def solve_problem(
    problem: str,
    data_class: str,
    solver_llm: str,
    rag_hint: str = None,
    coding_hint: str = None,
    logging_level=logging.INFO,
):
    """
    parameters:
    - problem: the problem to solve
    - data_class: the class of the problem, must be in ["algebra","counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    - solver_llm: the llm used to solve the problem
    - rag_hint: the hint (obtained from RAG) to solve the problem, default None
    - coding_hint: the hint (obtained from coding LLM) to solve the problem, default None

    TODO: add other parameters (e.g., temperature)

    return:
        the solution to the problem, a string
    """

    setup_logging(logging_level)

    instruct_query = f"""Imagine you are a math expert in {data_class}, diligently solving a math problem during an exam. You will be provided with a well-posed math problem in LaTeX format wrapped in three backticks (```), along with any available hints. Your task is to utilize all the information provided to deliver a comprehensive, detailed, and clear solution to the math problem, presented in LaTeX form. Additionally, you must render all calculation steps to demonstrate your reasoning and methodology. Your final sentence MUST be 'Therefore, the solution is XXX', where XXX represents the final solution obtained. Accuracy is paramount, with no mistakes permitted."""

    user_query = f"""Here is the problem you need to solve:\n```\n{problem}\n```\n"""

    if rag_hint is not None:
        user_query += f"""Here are some example problems and their solutions to help you better solve the problem I give you:\n```{rag_hint}```\n\n"""

    if coding_hint is not None:
        user_query += f"""We've also prepared Python scripts to address this problem. Upon execution, these codes yield `{coding_hint}`. While the output may not be entirely accurate, it could serve as a helpful tool to validate your answer. Exercise caution when utilizing this resource.\n"""

    logging.debug(f"query sent to {solver_llm}:\n", instruct_query + user_query)

    response = call_llm_api(
        model=solver_llm, system_query=instruct_query, user_query=user_query
    )
    logging.info("Problem solved successfully" if response else "Problem not solved")
    logging.debug(f"response from {solver_llm}:\n", response)

    return response


def solve_problem_by_coding(
    problem: str,
    data_class: str,
    solver_llm: str,
    max_attempt: int = 5,
    rag_hint: str = None,
    logging_level=logging.INFO,
):
    """
    parameters:
    - problem: the problem to solve
    - data_class: the class of the problem
    - coding_llm: the language model to use
    - max_attempt: the maximum number of attempts to execute the code if it fails
    - rag_hint: the hint (obtained from RAG) to solve the problem, default None

    return:
        the result of the code execution, if it succeeds in max_attempt attempts; otherwise None
    """

    setup_logging(logging_level)

    instruct_query = f"Imagine that you are an expert programmer that writes simple, concise code in Python. You will be given a mathematical problem in the domain of {data_class}, and your goal is to write some simple and executable python code to solve the problem. ONLY the code is needed, absolutely NO explanation. In case you still want to add explanation, please all explanations must be done in the form of COMMENTS in python. You are recommended to use SymPy for symbolic computation."

    user_query = (
        f"""Here is the problem you need to solve:\n\n```\n{problem}\n```\n\n"""
    )
    if rag_hint is not None:
        user_query += f"""Here are some example problems and their solutions to help you better solve the problem I give you:\n```{rag_hint}```\n"""

    response = call_llm_api(
        model=solver_llm, system_query=instruct_query, user_query=user_query
    )
    response = extract_code_blocks(response)
    response.replace("```", "")
    resp = remove_python_header(response)
    logging.debug(f"query sent to {solver_llm}:\n", instruct_query + user_query)
    logging.debug(f"response from {solver_llm}:\n", resp)

    # Attempt to execute the code and capture the output
    remaining_attempt = max_attempt
    code_result = None
    error = None
    while remaining_attempt > 0:
        succeed, code_result = exec_capture_stdout(resp)
        logging.info("code runs successfully" if succeed else "code failed to run")
        if succeed:
            break
        if remaining_attempt == 1:
            remaining_attempt -= 1
            continue
        error = f"Error in code execution: {code_result}"
        logging.error(f"{error}. Retrying...")
        user_query += f"\nYou provided this response.\n{resp}\nHowever, execution of the response resulted in an error.\n{error}\nPlease rectify the error and attempt again. ONLY the corrected code WITHOUT any error is required. NO explanations are needed within the code; however, if you choose to include any, they must be in the form of COMMENTS in Python. Using SymPy for symbolic computation is recommended.\n\n"
        logging.debug(f"query sent to {solver_llm}: ", instruct_query + user_query)
        response = call_llm_api(
            model=solver_llm, system_query=instruct_query, user_query=user_query
        )
        response = extract_code_blocks(response)
        response.replace("```", "")
        resp = remove_python_header(response)
        logging.debug(f"response from {solver_llm}:\n", resp)
        remaining_attempt -= 1
    if remaining_attempt == 0:
        logging.error(f"Consistent error in code execution: {code_result}. Will mark the problem as unsolved.")
        return
    else:
        return code_result
