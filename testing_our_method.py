from constants import ALL_PROBLEM_CLASSES
from solve_prob_end2end import solve_prob_end2end, solve_prob_directly
import os
import json
import random


def subsample(dataset, sample_size):
    assert isinstance(sample_size, int) and 0 <= sample_size <= len(
        dataset
    ), "Invalid number of problem"
    return random.sample(dataset, sample_size) if sample_size else dataset


def test_accuracy(
    data_class: str,
    num_problems: int,
    coding_llm: str,
    main_solver_llm: str,
    judging_llm: str,
    levels: list = [1, 2, 3, 4, 5],
    full_method: bool = True,
    show_details: bool = True,
):
    """
    Input:
        data_class: the class of the problem
        num_problems: the total number of problems to test, *before* seiving its level. If 0, test all problems.
        levels: the level of the problem
        full_method: True if use the end-to-end method (i.e. with RAG and code LLM),
                     False if use the direct method
        show_details: when True, print the details of the testing process
    Output:
        a list of accuracy ( = number of correctly-solved problems / number of all problems ) at each level
    """

    assert data_class in ALL_PROBLEM_CLASSES, "Invalid data class"
    assert all(
        [level in [1, 2, 3, 4, 5] for level in levels]
    ), "Levels must be in 1 to 5"

    prob_num = [0, 0, 0, 0, 0]
    # num of tested problems at each level
    correct_num = [0, 0, 0, 0, 0]
    # num of correctly-solved problems at each level

    dataset_path = os.path.join(
        os.getcwd(), "merged_dataset", "test", data_class, "merged.json"
    )
    dataset = json.load(open(dataset_path))
    assert isinstance(num_problems, int) and 0 <= num_problems <= len(
        dataset
    ), "Invalid number of problem"
    if num_problems:  # subsample num_problems problems
        dataset = random.sample(dataset, num_problems)

    if show_details:
        print(
            f"Testing on dataset '{data_class}' with levels {str(sorted(set(levels)))}."
        )

    for i in range(len(dataset)):
        level = dataset[i]["level"]
        if level not in levels:
            print(
                f"[{i}/{len(dataset)}] Problem {dataset[i]['filename']} (level {level}) is not in the test levels."
            )
            continue
        prob_num[level - 1] += 1
        if show_details:
            print(
                f"[{i}/{len(dataset)}] Problem {dataset[i]['filename']} (level {level}): ",
                end="",
            )

        prob = dataset[i]["problem"]
        sol = dataset[i]["solution"]
        try:
            is_correct = (
                solve_prob_end2end(
                    prob, data_class, sol, coding_llm, main_solver_llm, judging_llm
                )
                if full_method
                else solve_prob_directly(
                    prob, data_class, sol, main_solver_llm, judging_llm
                )
            )
            if show_details:
                print("correct" if is_correct else "incorrect")
            if is_correct:
                correct_num[level - 1] += 1

        except:
            if show_details:
                print("Failed in judging. This problem will not be counted.")
            prob_num[level - 1] -= 1

    return list(map(lambda x, y: x / y if y != 0 else 0, correct_num, prob_num))
    # accuracy at each level


if __name__ == "__main__":
    print(
        test_accuracy(
            data_class="geometry",
            num_problems=10,
            coding_llm="llama3-70b-8192",
            main_solver_llm="llama3-70b-8192",
            judging_llm="llama3-70b-8192",
            levels=[1, 2, 3, 4, 5],
            full_method=True,  # False
            show_details=True,
        )
    )
