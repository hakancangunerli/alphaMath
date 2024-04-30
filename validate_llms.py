"""
This file contains the function to validate the solver LLM.
"""

from judge_correctness import judge_correctness
from constants import ALL_PROBLEM_CLASSES
import re
from solver import solve_problem_by_coding
from logging_config import setup_logging
import logging

pattern = re.compile(r"\[asy\].*?\[/asy\]", re.DOTALL)


def remove_asy_tags(text):
    """
    Remove the [asy] tags from the solution.
    They are Asymptote code that are used for plotting, which are not relevant here."""
    global pattern
    return re.sub(pattern, "", text)


def validate_solver_llm(
    solve_method,
    data_class: str,
    dataset: list,
    solver_llm: str,
    levels: list = [1, 2, 3, 4, 5],
    judging_llm: str = "gpt-3.5-turbo",
    max_rejudge: int = 3,
    test_mode=False,
    logging_level=logging.INFO,
):
    """
    parameters:
    - solve_method: solve_problem_by_coding or solve_problem
    - data_class: the class of the data, must be in ['algebra','counting_and_probability','geometry','intermediate_algebra','number_theory','prealgebra','precalculus']
    - dataset: the dataset, a list of problems
    - solver_llm: the llm used to solve the problem or write the code
    - levels: the levels of the data, default is all 5 levels
    - judging_llm: the llm used to judge the correctness of the solution
    - max_rejudge: the maximum number of times to rejudge the solution if it fails

    return:
        a list of the accuracy at all levels, e.g., [.9, .8, .7, .6, .5]
    and a list of the failure rate at all levels, e.g., [.1, .2, .3, .4, .5]
    """

    setup_logging(logging_level)

    assert data_class in ALL_PROBLEM_CLASSES, "Invalid data class"
    assert all(
        [level in [1, 2, 3, 4, 5] for level in levels]
    ), "Levels must be in 1 to 5"

    logging.info(f"Testing dataset {data_class} with levels {str(sorted(set(levels)))}")

    prob_num = [0, 0, 0, 0, 0]  # num of tested problems at each level
    correct_num = [0, 0, 0, 0, 0]  # num of correctly-solved problems at each level
    failed_num = [
        0,
        0,
        0,
        0,
        0,
    ]  # num of failed files (i.e., can't generate the answer) at each level

    # # retry files that had errors in the first round
    # retry_files = []

    for i in range(len(dataset)):
        logging.info(f"Testing problem {dataset[i]['filename']} of {data_class}")

        level = dataset[i]["level"]
        if level not in levels:
            continue
        prob_num[level - 1] += 1

        prob = dataset[i]["problem"]
        sol = dataset[i]["solution"]
        ans = solve_method(prob, data_class, solver_llm, test_mode=test_mode)

        if not ans:
            logging.warning(f"Failed in solving problem {dataset[i]['filename']}.")
            failed_num[level - 1] += 1
            continue

        for _ in range(max_rejudge):
            try:
                is_correct = judge_correctness(prob, sol, ans, llm=judging_llm)
                if is_correct:
                    correct_num[level - 1] += 1
                break
            except ValueError:
                if _ < max_rejudge - 1:
                    continue
                else:
                    logging.warning(
                        f"Failed in judging correctness for problem {dataset[i]['filename']} for {max_rejudge} times. We will not count this problem."
                    )
                    prob_num[level - 1] -= 1

        logging.info(
            f"Problem numbers at each level attempted to solve so far: {prob_num}"
        )
        logging.info(
            f"Correctly-solved problem numbers at each level so far: {correct_num}"
        )
        logging.info(f"Failed problem numbers at each level so far: {failed_num}")

    return list(map(lambda x, y: x / y if y != 0 else 0, correct_num, prob_num)), list(
        map(lambda x, y: x / y if y != 0 else 0, failed_num, prob_num)
    )
    # accuracy at each level and failed rate at each level
