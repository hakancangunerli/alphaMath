import io
import re
import sys
from call_apis import *


def remove_python_header(code_output):
    # Remove any line that only contains 'python'
    code_output = re.sub(
        r"^python\s*$", "", code_output, flags=re.IGNORECASE | re.MULTILINE
    )
    return code_output.strip()


def exec_capture_stdout(code):
    # Function to execute code and capture stdout
    captured_output = io.StringIO()

    # Save the current stdout so we can restore it later
    original_stdout = sys.stdout
    sys.stdout = captured_output
    exec(code)
    sys.stdout = original_stdout
    return captured_output.getvalue()


def llm_solver(
    problem: str, data_class: str, solver_llm: str, hint: str = None, test_mode=False
):
    """
    parameters:
    - problem: the problem to solve
    - data_class: the class of the problem, must be in ["algebra","counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    - solver_llm: the llm used to solve the problem
    - hint: the hint (obtained from RAG / code / ...) to solve the problem, default None

    TODO: add other parameters (e.g., temperature)

    return:
        the solution to the problem
    """

    instruct_query = f"""Imagine you are a math expert in {data_class}, who is solving a math problem in all seriousness during the exam. You will be given a math problem in LaTeX format wrapped in three backticks (```), which is well-posed and has a unique solution. Your task is to provide a complete, detailed and clear solution to it in LaTeX form, as well as rendering all the calculation steps. Your last sentense MUST be 'Therefore, the solution is XXX', which outputs the final solution. Be extremely accurate. No mistakes allowed."""

    user_query = (
        f"""Here is the problem you need to solve:\n\n```\n{problem}\n```\n\n"""
    )

    if hint is not None:
        user_query += (
            f"""Let me give you some possibly useful hints:\n```{hint}```\n\n"""
        )

    if test_mode:
        print(f"query sent to {solver_llm}:\n", instruct_query + user_query)

    return call_llm_api(
        model=solver_llm, system_query=instruct_query, user_query=user_query
    )


def llm_coder(problem: str, data_class, coder_llm, test_mode=False):
    """
    parameters:
    - problem: the problem to solve
    - llm: the language model to use.
    """
    instruct_query = f"You are an expert programmer that writes simple, concise code in Python. You have been given mathematical problem in the domain of {data_class}, and you need to write a simple and executable python code to solve the problem. ONLY the code is needed, Absolutely NO explanation. Use SymPy."
    user_query = (
        f"""Here is the problem you need to solve:\n\n```\n{problem}\n```\n\n"""
    )
    response = call_llm_api(
        model=coder_llm, system_query=instruct_query, user_query=user_query
    )
    resp = response.replace("```", "")
    resp = remove_python_header(resp)
    if test_mode:
        print(f"query sent to {coder_llm}:\n", instruct_query + user_query)
        print(f"response from {coder_llm}:\n", resp)

    # Attempt to execute the code and capture the output
    attempt = 5
    code_result = None
    error = None
    while attempt > 0:
        try:
            code_result = exec_capture_stdout(resp)
            if test_mode:
                print(f"code result: {code_result}")
            break
        except Exception as e:
            error = e
            if attempt == 1:
                attempt -= 1
                continue
            error = f"Error in code execution: {e}"
            user_query += f"\nYou provided this response.\n{resp}\nHowever, execution of the response resulted in an error.\n{error}\nPlease correct the error and try again. ONLY the corrected code is needed without any bugs,Absolutely NO explanation. Use SymPy."
            if test_mode:
                print(f"query sent to {coder_llm}: ", instruct_query + user_query)
            response = call_llm_api(
                model=coder_llm, system_query=instruct_query, user_query=user_query
            )
            resp = response.replace("```", "")
            resp = remove_python_header(resp)
            if test_mode:
                print(f"response from {coder_llm}:\n", resp)
            attempt -= 1
    if attempt == 0:
        print(f"Error in code execution: {error}")
        return None
    return code_result
