from langchain.output_parsers import ResponseSchema

qa_generation_prompt = """<|im_start|>system
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

{schema}
<|im_end|>
<|im_start|>user
Context: {context}<|im_end|>
<|im_start|>assistant"""

qa_response_schema = [
    ResponseSchema(name="question", description="your factoid question in German"),
    ResponseSchema(
        name="answer", description="your answer to the factoid question in German"
    ),
]

bing_chat_template = """
You are an expert assistant specializing in answering questions about lamps. Your goal is to provide accurate, concise, and helpful responses based on the given context. Follow these guidelines when answering the user's question:

    - Review the provided context carefully to ensure you have information related to the question.
    - Keep your answer brief and to the point, limiting it to a maximum of 3 sentences.
    - If you cannot answer the question based on the given context, inform the user rather than inventing an answer.
    - If the context is incomplete or irrelevant, politely inform the user that the provided information is insufficient to answer their question.

Format your response according to the provided schema: {schema}

Context:
{context}

Question: 
{question}"""

bing_chat_response_schema = [
    ResponseSchema(name="answer", description="Answer to the question")
]
