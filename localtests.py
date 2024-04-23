import json
import os
import random
import unittest


class LocalTests(unittest.TestCase):
    def test_call_apis(self):
        from call_apis import call_llm_api

        for model in ["llama3-70b-8192", "gpt-3.5-turbo"]:
            resp = call_llm_api(
                model=model,
                system_query="Hi! I am a student who is trying to solve a math problem. Can you help me?",
                user_query="Let $\\mathbf{a} = \\begin{pmatrix} -3 \\\\ 10 \\\\ 1 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} 5 \\\\ \\pi \\\\ 0 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} -2 \\\\ -2 \\\\ 7 \\end{pmatrix}.$  Compute\n\\[(\\mathbf{a} - \\mathbf{b}) \\cdot [(\\mathbf{b} - \\mathbf{c}) \\times (\\mathbf{c} - \\mathbf{a})].\\]",
            )
            # check if the response is not empty and returns a string
            self.assertTrue(resp)
            self.assertIsInstance(resp, str)

    def test_validate_solver_llm(self):
        from validate_llms import validate_solver_llm
        from solver import llm_solver
        from constants import DEFAULT_JUDGE_LLM, DEFAULT_SOLVER_LLM

        dataset_path = os.path.join(
            os.getcwd(), "merged_dataset", "train", "algebra", "merged.json"
        )
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        subsampled_dataset = random.sample(dataset, 2)
        resp = validate_solver_llm(
            solver=llm_solver,
            data_class="algebra",
            dataset=subsampled_dataset,
            levels=[1, 2, 3, 4, 5],
            solver_llm=DEFAULT_SOLVER_LLM,
            judging_llm=DEFAULT_JUDGE_LLM,
        )
        # check if the response is not empty and returns a list
        self.assertTrue(resp)
        self.assertIsInstance(resp, list)

        # check if all the elements in the list are between 0 and 1
        for i in resp:
            self.assertTrue(0 <= i <= 1)

    def test_llm_solver(self):
        from solver import llm_solver

        resp = llm_solver(
            problem="What is the solution to the equation $x^2-4=0$?",
            data_class="algebra",
            solver_llm="llama3-70b-8192",
        )
        # check if the response is not empty and returns a string
        self.assertTrue(resp)
        self.assertIsInstance(resp, str)

    def test_llm_coder(self):
        from solver import llm_coder

        resp = llm_coder(
            problem="What is the solution to the equation $x^2-4=0$?",
            data_class="algebra",
            coder_llm="codegemma",
        )
        # check if the response is not empty and returns a string
        self.assertTrue(resp)
        self.assertIsInstance(resp, str)

    def test_judge_correctness(self):
        from judge_correctness import judge_correctness

        resp = judge_correctness(
            prob="Find the phase shift of the graph of $y = 2 \\sin \\left( 2x + \\frac{\\pi}{3} \\right).$",
            sol="Since the graph of $y = 2 \\sin \\left( 2x + \\frac{\\pi}{3} \\right)$ is the same as the graph of $y = 2 \\sin 2x$ shifted $\\frac{\\pi}{6}$ units to the left, the phase shift is $\\boxed{-\\frac{\\pi}{6}}.$\n\n[asy]import TrigMacros;\n\nsize(400);\n\nreal g(real x)\n{\n\treturn 2*sin(2*x + pi/3);\n}\n\nreal f(real x)\n{\n\treturn 2*sin(2*x);\n}\n\ndraw(graph(g,-2*pi,2*pi,n=700,join=operator ..),red);\ndraw(graph(f,-2*pi,2*pi,n=700,join=operator ..));\ntrig_axes(-2*pi,2*pi,-3,3,pi/2,1);\nlayer();\nrm_trig_labels(-4,4, 2);\n[/asy]",
            ans="$-pi/6$",
            llm="llama3-70b-8192",
        )
        # check if the response is true
        self.assertTrue(resp)


if __name__ == "__main__":
    unittest.main()
