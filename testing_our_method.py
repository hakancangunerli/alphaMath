from itertools import groupby
from constants import ALL_PROBLEM_CLASSES
from solve_prob_end2end import solve_prob_end2end
from logging_config import setup_logging
import logging
import os
import json
import random


def subsample(dataset, sample_size):
    assert isinstance(sample_size, int) and 0 <= sample_size <= len(dataset), "Invalid sample size"

    # Sort the dataset by 'level' and then group by 'level'.
    dataset_sorted = sorted(dataset, key=lambda x: x["level"])
    groups = [list(g) for _, g in groupby(dataset_sorted, key=lambda x: x["level"])]

    sampled_data = []
    total_groups = len(groups)

    # Calculate distribution per level based on the rules:
    distribution = []
    if sample_size <= total_groups:
        distribution = [1 if i < sample_size else 0 for i in range(total_groups)]
    else:
        base = sample_size // total_groups
        extra = sample_size % total_groups
        distribution = [base + (1 if i < extra else 0) for i in range(total_groups)]

    # Sample according to the distribution plan
    count = 1
    for dist, group in zip(distribution, groups):
        print(f"Level {count}: {dist} problem(s) sampled")
        if dist <= len(group):
            sampled_data.extend(random.sample(group, dist))
        else:
            sampled_data.extend(group)  # Take all if not enough to meet distribution
        count += 1

    return sampled_data

def test_accuracy(
    data_class: str,
    dataset: list,
    coding_llm: str,
    main_solver_llm: str,
    judging_llm: str,
    levels: list = [1, 2, 3, 4, 5],
    use_rag: bool = True,
    logging_level: str = logging.INFO,
):
    """
    Input:
        data_class: the class of the problem
        dataset: the dataset to test
        levels: the level of the problem
        full_method: True if use the end-to-end method (i.e. with RAG and code LLM),
                     False if use the direct method
        show_details: when True, print the details of the testing process
    Output:
        a list of accuracy ( = number of correctly-solved problems / number of all problems ) at each level
    """

    print(f"Testing on dataset '{data_class}' with levels {str(sorted(set(levels)))}.")

    setup_logging(logging_level)

    if coding_llm is None and main_solver_llm is None:
        logging.error("Both coding_llm and solving_llm are None.")
        raise Exception("Both coding_llm and solving_llm cannot be None.")

    assert data_class in ALL_PROBLEM_CLASSES, "Invalid data class"
    assert all(
        [level in [1, 2, 3, 4, 5] for level in levels]
    ), "Levels must be in 1 to 5"

    prob_num = [0, 0, 0, 0, 0]
    # num of tested problems at each level
    correct_num = [0, 0, 0, 0, 0]
    # num of correctly-solved problems at each level

    logging.info(
        f"Testing on dataset '{data_class}' with levels {str(sorted(set(levels)))}."
    )

    for i in range(len(dataset)):
        level = dataset[i]["level"]
        logging.info(
            f"Working on [{i+1}/{len(dataset)}] Problem {dataset[i]['filename']} (level {level})..."
        )
        if level not in levels:
            logging.info(
                f"[{i+1}/{len(dataset)}] Problem {dataset[i]['filename']} (level {level}) is not in the test levels."
            )
            continue
        prob_num[level - 1] += 1

        prob = dataset[i]["problem"]
        sol = dataset[i]["solution"]
        try:
            is_correct = (
                solve_prob_end2end(
                    problem=prob, data_class=data_class, correct_solution=sol, coding_llm=coding_llm, main_solver_llm=main_solver_llm, judging_llm=judging_llm, use_rag=use_rag, logging_level=logging_level
                )
            )
            logging.info(f"[{i+1}/{len(dataset)}] Problem {dataset[i]['filename']} (level {level}): {'correct' if is_correct else 'incorrect'}")
            if is_correct:
                correct_num[level - 1] += 1

        except Exception as e:
            logging.warning(f"Error in solving or judging: {e}. This problem will not be counted.")
            prob_num[level - 1] -= 1

    return list(map(lambda x, y: round(x / y, 2) if y != 0 else 0, correct_num, prob_num))
    # accuracy at each level


if __name__ == "__main__":
    # set the data class and the number of problems to test
    data_class = "geometry"
    num_problems = 10

    dataset_path = os.path.join(
        os.getcwd(), "merged_dataset", "test", data_class, "merged.json"
    )
    dataset = json.load(open(dataset_path))
    assert isinstance(num_problems, int) and 0 <= num_problems <= len(
        dataset
    ), "Invalid number of problem"
    if num_problems:  # subsample num_problems problems
        dataset = subsample(dataset, num_problems)
    print(
        test_accuracy(
            data_class="geometry",
            dataset=dataset,
            coding_llm="llama3-70b-8192",
            main_solver_llm="llama3-70b-8192",
            judging_llm="llama3-70b-8192",
            levels=[1, 2, 3, 4, 5],
            use_rag=True,
            logging_level=logging.INFO,
        )
    )
