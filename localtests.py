import json
import os
import random
import unittest
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

class LocalTests(unittest.TestCase):
    @unittest.skipIf(cfg["api_key"]["groq"] is "", "Open API key not set")
    def test_call_apis_groq(self):
        from call_apis import call_llm_api

        resp = call_llm_api(
            model="llama3-70b-8192",
            system_query="Hi! I am a student who is trying to solve a math problem. Can you help me?",
            user_query="Let $\\mathbf{a} = \\begin{pmatrix} -3 \\\\ 10 \\\\ 1 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} 5 \\\\ \\pi \\\\ 0 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} -2 \\\\ -2 \\\\ 7 \\end{pmatrix}.$  Compute\n\\[(\\mathbf{a} - \\mathbf{b}) \\cdot [(\\mathbf{b} - \\mathbf{c}) \\times (\\mathbf{c} - \\mathbf{a})].\\]",
        )
        # check if the response is not empty and returns a string
        self.assertTrue(resp)
        self.assertIsInstance(resp, str)

    @unittest.skipIf(cfg["api_key"]["openai"] is "", "Groq API key not set")
    def test_call_apis_openai(self):
        from call_apis import call_llm_api

        resp = call_llm_api(
            model="gpt-3.5-turbo",
            system_query="Hi! I am a student who is trying to solve a math problem. Can you help me?",
            user_query="Let $\\mathbf{a} = \\begin{pmatrix} -3 \\\\ 10 \\\\ 1 \\end{pmatrix},$ $\\mathbf{b} = \\begin{pmatrix} 5 \\\\ \\pi \\\\ 0 \\end{pmatrix},$ and $\\mathbf{c} = \\begin{pmatrix} -2 \\\\ -2 \\\\ 7 \\end{pmatrix}.$  Compute\n\\[(\\mathbf{a} - \\mathbf{b}) \\cdot [(\\mathbf{b} - \\mathbf{c}) \\times (\\mathbf{c} - \\mathbf{a})].\\]",
        )
        # check if the response is not empty and returns a string
        self.assertTrue(resp)
        self.assertIsInstance(resp, str)

    def test_validate_solver_llm(self):
        from validate_llms import validate_solver_llm
        from solver import solve_problem
        from constants import DEFAULT_JUDGE_LLM, DEFAULT_SOLVER_LLM

        dataset_path = os.path.join(
            os.getcwd(), "merged_dataset", "train", "algebra", "merged.json"
        )
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
        subsampled_dataset = random.sample(dataset, 2)
        resp = validate_solver_llm(
            solve_method=solve_problem,
            data_class="algebra",
            dataset=subsampled_dataset,
            levels=[1, 2, 3, 4, 5],
            solver_llm=DEFAULT_SOLVER_LLM,
            judging_llm=DEFAULT_JUDGE_LLM,
        )
        # check if the response is not empty and returns a tuple
        self.assertTrue(resp)
        self.assertIsInstance(resp, tuple)

        # check if the first element in the tuple is a list
        self.assertIsInstance(resp[0], list)

        # check if the second element in the tuple is a list
        self.assertIsInstance(resp[1], list)

        # check if all the elements in the first list are between 0 and 1
        for i in resp[0]:
            self.assertTrue(0 <= i <= 1)

        # check if all the elements in the second list are between 0 and 1
        for i in resp[1]:
            self.assertTrue(0 <= i <= 1)

    def test_llm_solver(self):
        from solver import solve_problem

        resp = solve_problem(
            problem="What is the solution to the equation $x^2-4=0$?",
            data_class="algebra",
            solver_llm="llama3-70b-8192",
        )
        # check if the response is not empty and returns a string
        self.assertTrue(resp)
        self.assertIsInstance(resp, str)

    def test_llm_coder(self):
        from solver import solve_problem_by_coding

        resp = solve_problem_by_coding(
            problem="What is the solution to the equation $x^2-4=0$?",
            data_class="algebra",
            solver_llm="codegemma",
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

    def test_rag_setup(self):
        from db_setup import load_rag, test_rag

        try:
            load_rag()
            test_rag()
        except:
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
