"""
Judge the correctness of a student's answer to a math problem.
"""

from call_apis import *
from logging_config import setup_logging
import logging


prompt_examples = [
    # these examples come from the training set of MATH
    # we will use the testing set to evaluate the performance
    {
        "prob": "A point $(x,y)$ is randomly picked from inside the rectangle with vertices  $(0,0)$, $(3,0)$, $(3,2)$, and $(0,2)$.  What is the probability that  $x < y$?",
        "sol": "The point $(x,y)$ satisfies $x < y$ if and only if it belongs to the shaded triangle bounded by the lines $x=y$, $y=2$, and $x=0$, the area of which is 2.  The rectangle has area 6, so the probability in question is $\\dfrac{2}{6} = \\boxed{\\dfrac{1}{3}}$.]",
        "ans": "1/3",
        "correct": True,
    },
    {
        "prob": "Simplify $\\tan \\frac{\\pi}{24} + \\tan \\frac{7 \\pi}{24}.$",
        "sol": "We can write\n\\[\\tan \\frac{\\pi}{24} + \\tan \\frac{7 \\pi}{24} = \\frac{\\sin \\frac{\\pi}{24}}{\\cos \\frac{\\pi}{24}} + \\frac{\\sin \\frac{7 \\pi}{24}}{\\cos \\frac{7 \\pi}{24}} \n= \\frac{\\sin \\frac{\\pi}{24} \\cos \\frac{7 \\pi}{24} + \\cos \\frac{\\pi}{24} \\sin \\frac{7 \\pi}{24}}{\\cos \\frac{\\pi}{24} \\cos \\frac{7 \\pi}{24}}.\\]By the angle addition formula and the product-to-sum formula,\n\\begin{align*}\n\\frac{\\sin \\frac{\\pi}{24} \\cos \\frac{7 \\pi}{24} + \\cos \\frac{\\pi}{24} \\sin \\frac{7 \\pi}{24}}{\\cos \\frac{\\pi}{24} \\cos \\frac{7 \\pi}{24}} &= \\frac{\\sin (\\frac{\\pi}{24} + \\frac{7 \\pi}{24})}{\\frac{1}{2} (\\cos \\frac{\\pi}{3} + \\cos \\frac{\\pi}{4})} \\\\\n&= \\frac{2 \\sin \\frac{\\pi}{3}}{\\cos \\frac{\\pi}{3} + \\cos \\frac{\\pi}{4}} \\\\\n&= \\frac{\\sqrt{3}}{\\frac{1}{2} + \\frac{\\sqrt{2}}{2}} \\\\\n&= \\frac{2 \\sqrt{3}}{1 + \\sqrt{2}} \\\\\n&= \\frac{2 \\sqrt{3} (\\sqrt{2} - 1)}{(\\sqrt{2} + 1)(\\sqrt{2} - 1)} \\\\\n&= \\boxed{2 \\sqrt{6} - 2 \\sqrt{3}}.\n\\end{align*}",
        "ans": "2*sqrt(2)*(sqrt(3)-1)",
        "correct": False,
    },
    {
        "prob": "For real numbers $a,$ $b,$ and $c,$ the matrix\n\\[\\begin{pmatrix} a & b & c \\\\ b & c & a \\\\ c & a & b \\end{pmatrix}\\]is not invertible.  List all possible values of\n\\[\\frac{a}{b + c} + \\frac{b}{a + c} + \\frac{c}{a + b}.\\]",
        "sol": "Since the matrix is not invertible, its determinant is 0, i.e.\n\\[\\begin{vmatrix} a & b & c \\\\ b & c & a \\\\ c & a & b \\end{vmatrix} = 0.\\]The determinant expands as\n\\begin{align*}\n\\begin{vmatrix} a & b & c \\\\ b & c & a \\\\ c & a & b \\end{vmatrix} &= a \\begin{vmatrix} c & a \\\\ a & b \\end{vmatrix} - b \\begin{vmatrix} b & a \\\\ c & b \\end{vmatrix} + c \\begin{vmatrix} b & c \\\\ c & a \\end{vmatrix} \\\\\n&= a(bc - a^2) - b(b^2 - ac) + c(ab - c^2) \\\\\n&= 3abc - a^3 - b^3 - c^3.\n\\end{align*}This factors as\n\\[3abc - a^3 - b^3 - c^3 = -(a + b + c)(a^2 + b^2 + c^2 - ab - ac - bc),\\]so either $a + b + c = 0$ or $a^2 + b^2 + c^2 - ab - ac - bc = 0.$\n\nIf $a + b + c = 0,$ then\n\\[\\frac{a}{b + c} + \\frac{b}{a + c} + \\frac{c}{a + b} = \\frac{a}{-a} + \\frac{b}{-b} + \\frac{c}{-c} = -3.\\]Now, suppose $a^2 + b^2 + c^2 - ab - ac - bc = 0.$  Then\n\\begin{align*}\n(a - b)^2 + (a - c)^2 + (b - c)^2 &= (a^2 - 2ab + b^2) + (a^2 - 2ac + c^2) + (b^2 - 2bc + c^2) \\\\\n&= 2(a^2 + b^2 + c^2 - ab - ac - bc) \\\\\n&= 0.\n\\end{align*}This forces $a = b = c,$ so\n\\[\\frac{a}{b + c} + \\frac{b}{a + c} + \\frac{c}{a + b} = \\frac{3}{2}.\\]Thus, the possible values of\n\\[\\frac{a}{b + c} + \\frac{b}{a + c} + \\frac{c}{a + b}\\]are $\\boxed{\\frac{3}{2}}$ and $\\boxed{-3}.$",
        "ans": "-3 and 1.5",
        "correct": True,
    },
]


def judge_correctness(
    prob: str,
    sol: str,
    ans: str,
    prompt_examples: str = prompt_examples,
    llm: str = "llama3-70b-8192",
    logging_level: str = logging.INFO,
):
    """
    parameters:
    - prob: the problem
    - sol: the solution
    - ans: the student's answer
    - prompt_examples: a list of dictionaries, each containing a problem, a solution, the student's answer, and whether the student's answer is correct
    - llm: the language model to use.

    return:
        True if the student's answer is correct, False otherwise
        If the LLM's response is invalid, raise an error
    """

    setup_logging(logging_level)

    instruct_query = """
Imagine that you are a high school math teacher correcting students' homework. You will be given three things: (1) a math problem in LaTeX, (2) a complete solution to this problem, in which the correct answer is wrapped in a box, i.e., \\boxed{answer} (note that there may be multiple boxes), and (3) the student's answer to this problem. These three items will all be wrapped in three backticks (```). Your task is to determine whether the student's answer is correct. If it is, you should respond with "Yes"; otherwise, you should respond with "No". DO NOT return anything other than "Yes" or "No".

The student's answer is considered correct if it is mathematically equivalent to the final answer in the solution. The correct answer can be a real number, a symbolic formula, or any other mathematical items (e.g., a set, an interval, etc.). For example, if the final answer in the solution is $\\boxed{4}$, then the student's answer is considered correct if it is $4$, $4.0$, $4.00$, $\\frac{8}{2}$, etc, and is considered incorrect if it is $4.1$, $4.01$, $\\frac{1}{4}$, etc; if the final answer in the solution is $\\boxed{4x+5}$, then the student's answer is considered correct if it is $4*x+5$, $4\\times x+5.0$, $(8x+10)/2$, etc, and is considered incorrect if it is $4x+5.1$, $4y+5$, $4X+5$, etc.

Here are some examples:
"""

    for i, ex in enumerate(prompt_examples, start=1):
        instruct_query += f"Example {i}:\nQ: (1) Problem: ```{ex['prob']}``` (2) Solution: ```{ex['sol']}``` (3) Student's answer: ```{ex['ans']}```.\n A: {'Yes' if ex['correct'] else 'No'}\n\n"

    user_query = (
        "Now, I want you to determine whether the student's answer is correct for the following problem.  Please make sure to ONLY answer in `YES` or `NO` without any mistake!\nQ: (1) Problem: ```"
        + prob
        + "``` (2) Solution: ```"
        + sol
        + "``` (3) Student's answer: ```"
        + ans
        + "```.\nA: "
    )

    logging.debug(f"Code for Judging correctness: {instruct_query}\n{user_query}")

    response = call_llm_api(
        model=llm, system_query=instruct_query, user_query=user_query
    ).lower()

    logging.debug(f"Response from the Judge LLM: {response}")

    if "yes" in response and "no" not in response:
        return True
    elif "no" in response and "yes" not in response:
        return False
    else:
        raise ValueError("Invalid response from the LLM")
