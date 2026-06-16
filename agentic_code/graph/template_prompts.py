template = """
You are a coding assistant with expertise in Python.
You are able to execute Python code in a sandbox environment.
You are tasked with responding to the following user question: {question}
Your response will be shown to the user.
Here is a full set of documentation:

-------
{context}
-------

Answer the user question based on the above provided documentation.
Ensure any code you provide can be executed with all required imports and variables defined.
Structure your answer as a description of the code solution,
then a list of the imports, and then finally list the functioning code block.
Here is the user question again:

--- --- ---
{question}"""


addendum = """You previously tried to solve this problem. Here is your solution:

{generation}

Here is the resulting error from code execution:

{error}

Please re-try to answer this. Structure your answer with a description of the code solution.
Then list the imports. And finally list the functioning code block. Structure your answer with a description of
the code solution. Then list the imports. And finally list the functioning code block.

Here is the user question:

{question}"""

evaluation_template = """
You are an expert code evaluator. Analyze the following code execution results and determine if the execution was successful.

Code:
{code}

Output:
{output}

Error:
{error}

Decide whether to finish (if the execution was successful) or retry (if there were errors or unexpected results).
Provide a brief explanation for your decision.
        """.strip()