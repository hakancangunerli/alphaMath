import io
import re
import sys
from call_apis import *


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


def solve_problem_by_coding(
    problem: str,
    data_class: str,
    solver_llm: str,
    max_attempt: int = 5,
    test_mode=False,
):
    """
    parameters:
    - problem: the problem to solve
    - data_class: the class of the problem
    - coding_llm: the language model to use
    - max_attempt: the maximum number of attempts to execute the code if it fails

    return:
        the result of the code execution, if it succeeds in max_attempt attempts; otherwise None
    """

    instruct_query = f"Imagine that you are an expert programmer that writes simple, concise code in Python. You will be given a mathematical problem in the domain of {data_class}, and your goal is to write some simple and executable python code to solve the problem. ONLY the code is needed, Absolutely NO explanation. In case you still want to add explanation, please all explanations must be done in the form of COMMENTS in python. You are recommended to use SymPy for symbolic computation."

    user_query = (
        f"""Here is the problem you need to solve:\n\n```\n{problem}\n```\n\n"""
    )
    response = call_llm_api(
        model=solver_llm, system_query=instruct_query, user_query=user_query
    )
    response = extract_code_blocks(response)
    response.replace("```", "")
    resp = remove_python_header(response)
    if test_mode:
        print(f"query sent to {solver_llm}:\n", instruct_query + user_query)
        print(f"response from {solver_llm}:\n", resp)

    # Attempt to execute the code and capture the output
    remaining_attempt = max_attempt
    code_result = None
    error = None
    while remaining_attempt > 0:
        succeed, code_result = exec_capture_stdout(resp)
        if test_mode:
            print("code runs successfully" if succeed else "code failed to run")
        if succeed:
            break
        if remaining_attempt == 1:
            remaining_attempt -= 1
            continue
        error = f"Error in code execution: {code_result}"
        user_query += f"\nYou provided this response.\n{resp}\nHowever, execution of the response resulted in an error.\n{error}\nPlease correct the error and try again. ONLY the corrected code is needed without any bugs. Absolutely NO explanation. In case you still want to add explanation, please all explanations must be done in the form of COMMENTS in python. You are recommended to use SymPy for symbolic computation.\n\n"
        if test_mode:
            print(f"query sent to {solver_llm}: ", instruct_query + user_query)
        response = call_llm_api(
            model=solver_llm, system_query=instruct_query, user_query=user_query
        )
        response = extract_code_blocks(response)
        response.replace("```", "")
        resp = remove_python_header(response)
        if test_mode:
            print(f"response from {solver_llm}:\n", resp)
        remaining_attempt -= 1
    if remaining_attempt == 0:
        print(error)
        return
    else:
        return code_result
