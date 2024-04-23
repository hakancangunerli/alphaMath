import json
import os
import random
from solver import llm_solver
from judge_correctness import judge_correctness
from constants import DEFAULT_JUDGE_LLM, ALL_PROBLEM_CLASSES
import re

pattern = re.compile(r"\[asy\].*?\[/asy\]", re.DOTALL)


def remove_asy_tags(text):
    """
    Remove the [asy] tags from the solution.
    They are Asymptote code that are used for plotting, which are not relevant here."""
    global pattern
    return re.sub(pattern, "", text)


def validate_solver_llm(
    solver,
    data_class: str,
    dataset: list,
    solver_llm=None,
    levels: list = [1, 2, 3, 4, 5],
    judging_llm: str = "gpt-3.5-turbo",
    test_mode=False,
):
    """
    parameters:
    - solver: the solver to be tested, returns a **string** of the answer
    - data_class: the class of the data, must be in ['algebra','counting_and_probability','geometry','intermediate_algebra','number_theory','prealgebra','precalculus']
    - args_solver: the arguments for the solver (if any)
    - levels: the levels of the data, default is all 5 levels
    - target_dir: the directory of the data (TODO: change to your own!)

    return:
    a list of the accuracy at all levels, whose first position is dummy. e.g., [0,0.9,0.8,0.7,0.6,0.5]
    """
    assert data_class in ALL_PROBLEM_CLASSES, "Invalid data class"
    assert all(
        [level in [1, 2, 3, 4, 5] for level in levels]
    ), "Levels must be in 1 to 5"

    print(f"Testing dataset {data_class} with levels {str(sorted(set(levels)))}")

    prob_num = [0, 0, 0, 0, 0]  # num of problems at each level
    correct_num = [0, 0, 0, 0, 0]  # num of correctly-solved problems at each level

    # retry files that had errors in the first round
    retry_files = []

    for i in range(len(dataset)):
        if test_mode:
            print(f"Testing problem {dataset[i]['filename']} of {data_class}")

        level = dataset[i]["level"]
        if level not in levels:
            continue
        prob_num[level - 1] += 1
        prob = dataset[i]["problem"]
        sol = remove_asy_tags(dataset[i]["solution"])
        ans = solver(prob, data_class, solver_llm)

        try:
            is_correct = judge_correctness(prob, sol, ans, llm=judging_llm)
            if is_correct:
                correct_num[level - 1] += 1
        except ValueError:
            print(
                f"Error in judging correctness for problem {dataset[i]['filename']}. Will retry later."
            )
            retry_files.append(i)
            prob_num[level - 1] -= 1

        if test_mode:
            print(
                f"Problem numbers at each level attempted to solve so far: {prob_num}"
            )
            print(
                f"Correctly-solved problem numbers at each level so far: {correct_num}"
            )

    # Retry the files that had errors
    if len(retry_files) > 0:
        print(f"Retrying {len(retry_files)} files")
        for i in retry_files:
            if test_mode:
                print(f"Again testing problem {dataset[i]['filename']} of {data_class}")

            level = dataset[i]["level"]
            prob_num[level - 1] += 1
            prob = dataset[i]["problem"]
            sol = remove_asy_tags(dataset[i]["solution"])
            try:
                is_correct = judge_correctness(prob, sol, ans, llm=judging_llm)
                if is_correct:
                    correct_num[level - 1] += 1
            except ValueError:
                print(
                    f"Error in judging correctness for problem {dataset[i]['filename']}"
                )
                prob_num[level - 1] -= 1

        if test_mode:
            print(
                f"Problem numbers at each level attempted to solve so far: {prob_num}"
            )
            print(
                f"Correctly-solved problem numbers at each level so far: {correct_num}"
            )

    return list(map(lambda x, y: x / y if y != 0 else 0, correct_num, prob_num))
