

SUB_QUERY_PROMPT = """
To answer this question more comprehensively, please break down the original question into up to four sub-questions that can run in parallel. Return as list of str.
If this is a very simple question and no decomposition is necessary, then keep the only one original question in the list.

Original Question: {original_query}

<EXAMPLE>
Example input:
"Explain deep learning"

Example output:
[
    "What is deep learning?",
    "What is the difference between deep learning and machine learning?",
    "What is the history of deep learning?"
]
</EXAMPLE>

Provide your response in list of str format:
"""

REFLECTION_PROMPT = """
Determine whether additional search queries are needed based on the original query, previous sub queries, and all retrieved document chunks. If further research is required, provide a Python list of up to 3 search queries. If no further research is required, return an empty list.
If the original query is to write a report, then you prefer to generate some further queries, instead return an empty list.

Original Query: {original_query}

Previous Sub Queries: {sub_queries}

And attached images.

Respond exclusively in valid List of str format without any other text.
"""