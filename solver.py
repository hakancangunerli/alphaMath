from call_apis import *


def llm_solver(problem: str, data_class, solver_llm):
    """
    parameters:
    - problem: the problem to solve
    - llm: the language model to use.

    TODO: add other parameters (e.g., temperature)

    return:
        the solution to the problem
    """

    instruct_query = f"""Imagine you are a math expert in {data_class}, who is solving a math problem in all seriousness during the exam. You will be given a math problem in LaTeX format wrapped in three backticks (```), which is well-posed and has a unique solution. Your task is to provide a complete, detailed and clear solution to it in LaTeX form, as well as rendering all the calculation steps. Your last sentense MUST be 'Therefore, the solution is XXX', which outputs the final solution. Be extremely accurate. No mistakes allowed."""

    user_query = (
        f"""Here is the problem you need to solve:\n\n```\n{problem}\n```\n\n"""
    )

    return call_llm_api(
        model=solver_llm, system_query=instruct_query, user_query=user_query
    )
