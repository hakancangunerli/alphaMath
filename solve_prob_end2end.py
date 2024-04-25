from judge_correctness import judge_correctness
from solver import solve_problem_by_coding, solve_problem
from db_setup import get_rag

rag_data_base = get_rag()


def solve_prob_end2end(
    problem: str,
    data_class: str,
    correct_solution: str,
    coding_llm: str,
    main_solver_llm: str,
    judging_llm: str,
    coding_max_attempt: int = 5,
    max_rejudge: int = 3,
    test_mode: bool = False,
):
    """
    The end to end process to solve a problem:
    1. Retrieve information from RAG
    2. Solve the problem by coding
    3. Solve the problem by the main LLM using the hint from RAG and the result from coding

    Output:
        True if the problem has been solved correctly,
        False if the problem has not been solved correctly,
        ValueError if the judging LLM's response is invalid for max_rejudge times
    """

    # query the rag system
    rag_items = dict()
    for doc in rag_data_base.similarity_search(problem):
        rag_items[doc.page_content] = doc.metadata["solution"]
    if not rag_items:
        rag_result = None
    else:
        rag_result = []
        for idx, (k, v) in enumerate(rag_items.items(), 1):
            rag_result.append(f'Example problem {idx}: {k}\n{"Solution: "+v}')
        rag_result = "\n".join(rag_result)
    if test_mode:
        print("Information retrieved from RAG: ```{rag_result}```")

    # solve the problem by coding
    coding_res = solve_problem_by_coding(
        problem,
        data_class,
        coding_llm,
        max_attempt=coding_max_attempt,
        rag_hint=rag_result,
        test_mode=test_mode,
    )
    if test_mode:
        print("The result from coding:", coding_res)

    # solve the problem by the main LLM
    main_solver_res = solve_problem(
        problem,
        data_class,
        main_solver_llm,
        rag_hint=rag_result,
        coding_hint=coding_res,
        test_mode=test_mode,
    )
    if test_mode:
        print("The result from the main solver:", main_solver_res)

    # validate the correctness of the results, judge the correctness for max_rejudge times
    for _ in range(max_rejudge):
        try:
            return judge_correctness(
                problem, correct_solution, main_solver_res, llm=judging_llm
            )
        except ValueError:
            continue
    
    raise ValueError(
        "Failed in judging correctness for problem for {max_rejudge} times. We will not count this problem."
    )


def solve_prob_directly(
    problem: str,
    data_class: str,
    correct_solution: str,
    main_solver_llm: str,
    judging_llm: str,
    max_rejudge: int = 3,
    test_mode: bool = False,
):
    """
    Solve a problem directly by the main_solver_llm, i.e., without using RAG or code LLM.

    Output:
        True if the problem has been solved correctly,
        False if the problem has not been solved correctly,
        ValueError if the judging LLM's response is invalid
    """

    # solve the problem by the main LLM
    main_solver_res = solve_problem(
        problem,
        data_class,
        main_solver_llm,
        test_mode=test_mode,
    )
    if test_mode:
        print("The result from the main solver:", main_solver_res)

    # validate the correctness of the results, judge the correctness for max_rejudge times
    for _ in range(max_rejudge):
        try:
            return judge_correctness(
                problem, correct_solution, main_solver_res, llm=judging_llm
            )
        except ValueError:
            continue
    raise ValueError


if __name__ == "__main__":
    print(
        solve_prob_end2end(
            problem="What is the integral of $f(x)=e^{-x^2/2}$ on R?",
            data_class="pre-calculus",
            correct_solution="$\sqrt{2\pi}$",
            coding_llm="llama3-70b-8192",
            main_solver_llm="llama3-70b-8192",
            judging_llm="llama3-70b-8192",
        )
    )

    print(
        solve_prob_directly(
            problem="What is the integral of $f(x)=e^{-x^2/2}$ on R?",
            data_class="pre-calculus",
            correct_solution="$\sqrt{2\pi}$",
            main_solver_llm="llama3-70b-8192",
            judging_llm="llama3-70b-8192",
        )
    )
