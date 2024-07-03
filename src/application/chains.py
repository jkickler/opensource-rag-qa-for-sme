import json
from pprint import pformat
from typing import Union, List

import application.templates as tl
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from utils.logging_utils import logger
from langchain_community.llms import LlamaCpp
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever


def log_execute(func: callable) -> callable:
    """Logs the execution of a chain with the given query."""

    def wrapper(chain, query, **kwargs):
        logger.info(f"EXECUTING {chain.__class__.__name__} WITH QUERY: {query}")
        result = func(chain, query, **kwargs)
        logger.info(f"RESULT:\n{pformat(result, sort_dicts=False)}")
        return result

    return wrapper


class Chain:
    """Base class for all chains in the application. Contains common methods and attributes."""

    def __init__(self):
        self.response_schema = tl.standard_response_schema
        self.prompt_template = tl.standard_prompt_template
        self.parser = self.create_parser()
        self.prompt = self.create_prompt()

    def sort_llm_output(self, dict: dict) -> dict:
        """Sorts the LLM output dictionary according to the key order."""
        key_order = [
            "question_id",
            "question_type",
            "solved",
            "question",
            "answer",
            "context",
        ]
        return {k: dict.get(k, None) for k in key_order}

    def to_bool(self, value):
        """Converts a string value to a boolean."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() == "true"
        raise ValueError(f"Invalid value for boolean conversion: {value}")

    def create_parser(self) -> StructuredOutputParser:
        """Creates a structured output parser from the response schema."""
        return StructuredOutputParser.from_response_schemas(self.response_schema)

    def create_prompt(self) -> PromptTemplate:
        """Creates a prompt from the prompt template and the response schema."""
        format_instructions = self.parser.get_format_instructions()
        return PromptTemplate.from_template(
            self.prompt_template,
            partial_variables={"schema": format_instructions},
        )


class ProductChain(Chain):
    """Chain for product-related queries."""

    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self.response_schema = tl.product_response_schema
        self.prompt_template = tl.product_prompt_template
        self.parser = self.create_parser()
        self.prompt = self.create_prompt()
        self.question_id = None

    def convert_llm_output(
        self, llm_output: Union[str, dict], product_info: dict
    ) -> dict:
        """
        Converts the LLM output to a standardized format.

        Args:
            llm_output (Union[str, dict]): The LLM output to be converted. It can be either a JSON string or a dictionary.
            product_info (dict): The product information to be included in the converted output.

        Returns:
            dict: The converted LLM output in sorted format.
        """
        llm_output = (
            json.loads(llm_output) if isinstance(llm_output, str) else llm_output
        )
        llm_output["context"] = product_info
        llm_output["question_type"] = "PRODUCT"
        llm_output["question_id"] = f"P{self.question_id}"
        return self.sort_llm_output(llm_output)

    @log_execute
    def execute(self, query: str, product_info: dict) -> dict:
        """
        Executes the chain for the given query and product information.

        Args:
            query (str): The query string.
            product_info (dict): Context about the product.

        Returns:
            dict: The response from the chain.
        """
        product_chain = self.prompt | self.llm
        response = product_chain.invoke({"question": query, "context": product_info})
        response = self.convert_llm_output(response, product_info)
        return response


class DocumentChain(Chain):
    """Chain for document retrieval queries."""

    def __init__(self, retriever: BaseRetriever, llm: LlamaCpp):
        self.llm = llm
        self.retriever = retriever
        self.response_schema = tl.document_reponse_schema
        self.prompt_template = tl.document_prompt_template
        self.parser = self.create_parser()
        self.prompt = self.create_prompt()
        self.question_id = None

    def concat_docs(self, docs: List[Document]) -> str:
        """Concatenates the page content of the documents to a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def convert_llm_output(self, llm_output: dict):
        """
        Converts the LLM output to a standardized format.

        Args:
            llm_output (dict): The LLM output to be converted.

        Returns:
            dict or None: The converted LLM output in sorted format, or None if an error occurred.
        """
        try:
            llm_output["context"] = [doc.dict() for doc in llm_output["context"]]
            llm_output["question_type"] = "DOCUMENT"

            llm_output_dict = json.loads(llm_output["llm_output"])
            llm_output_dict = {k.lower(): v for k, v in llm_output_dict.items()}

            # Sometimes the LLM output contains "antwort" and sometimes "answer",
            # so we need to check for both.
            answer_key = "antwort" if "antwort" in llm_output_dict else "answer"
            llm_output["answer"] = llm_output_dict[answer_key]
            del llm_output["llm_output"]
            llm_output["question_id"] = f"D{self.question_id}"

            return self.sort_llm_output(llm_output)
        except Exception as e:
            logger.error(
                f"Error occurred during conversion of LLM output: {e}\n {pformat(llm_output, sort_dicts=False)} "
            )
            return None

    @log_execute
    def execute(self, query: str) -> dict:
        """
        Executes the chain for the given query and product information.

        Args:
            query (str): The query string.

        Returns:
            dict: The response from the chain.
        """
        doc_chain_from_docs = (
            RunnablePassthrough.assign(
                context=(lambda x: self.concat_docs(x["context"]))
            )
            | self.prompt
            | self.llm
        )
        doc_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(llm_output=doc_chain_from_docs)
        response = doc_chain_with_source.invoke(query)
        response = self.convert_llm_output(response)
        return response


class Judge(Chain):
    def __init__(self, llm: LlamaCpp):
        self.llm = llm
        self.llm_response = None
        self.response_schema = tl.judge_schema
        self.prompt_template = tl.judge_prompt_template
        self.parser = None
        self.prompt = self.create_prompt()
        self.label = None

    def create_prompt(self) -> PromptTemplate:
        """Creates a prompt from the prompt template and the response schema."""
        return PromptTemplate.from_template(
            self.prompt_template,
            partial_variables={"schema": self.response_schema},
        )

    def judge_output(self) -> dict:
        """Generates the correctness score and justification for the LLM output.

        Returns:
            dict: The judged output with correctness score and justification.
        """
        logger.info(f"EXECUTING {self.__class__.__name__}")

        judge_chain = self.prompt | self.llm
        response = judge_chain.invoke(self.prepare_input())
        response_dict = {**json.loads(response), **self.llm_response}

        logger.info(f"RESULT:\n{pformat(response_dict, sort_dicts=False)}")
        return response_dict

    def prepare_input(self) -> dict:
        """Prepares the input for the judge chain."""
        return {
            "question": self.llm_response["question"],
            "answer": self.llm_response["answer"],
            "context": self.llm_response["context"],
        }

    def check_solved(self) -> bool:
        """Checks if the LLM response is solved."""
        return self.to_bool(self.llm_response["solved"])

    def extract_page_content(self) -> List[str]:
        """Removes the metadata from the LLM response and returns the page content"""
        return [item["page_content"] for item in self.llm_response["context"]]

    def execute(self, llm_response: dict) -> dict:
        """Evaluates the given LLM response.

        By a PRODUCT question, the 'solved' key is checked and the response is returned if solved.
        By a DOCUMENT question, the correctness score is checked and the response is returned if the score is less then 3.

        Args:
            llm_response (dict): The LLM response to evaluate.

        Returns:
            dict: The evaluated response.
        """

        self.llm_response = llm_response
        response = None

        # PRODUCT BLOCK
        if self.llm_response["question_type"] == "PRODUCT":
            if not self.check_solved():
                response = self.llm_response
            else:
                print(f"{self.llm_response['answer']}\n")
        # DOCUMENT BLOCK
        elif self.llm_response["question_type"] == "DOCUMENT":
            self.llm_response["context"] = self.extract_page_content()
            response = self.judge_output()
            response = self.check_correctness(response)

        return response

    def check_correctness(self, response: dict):
        """
        Checks the correctness of the response and prompts the user for feedback if necessary.

        Args:
            response (dict): The response dictionary.

        Returns:
            dict: The modified response dictionary or None if the user is satisfied with the answer.
        """
        if response.get("correctness", None) is None or response["correctness"] < 3:
            return response

        if self.label == "chat":
            print(f"{response['answer']}\n")
            user_question = input(
                ">>> Sind Sie zufrieden mit der Antwort? (Ja/Nein):\n"
            )
            # Returns None if user is satisfied with the answer, so comm_handler is not called later.
            return response if user_question.lower() == "nein" else None
        else:
            return response
