from pprint import pformat
import re
import os
import json
import application.templates as tl
from utils.logging_utils import logger
from application.knowledge_base import KnowledgeBase


class CommunicationHandler:
    """Handles communication with application components and human experts."""

    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initializes a CommunicationHandler object.

        Args:
            knowledge_base (KnowledgeBase): The knowledge base.
        """
        self.email_template = tl.email_template
        self.kb = knowledge_base
        self.email_storage = self.kb.path_email_storage

    def send_email_and_save_json(self, email: str, llm_output: dict) -> None:
        """
        Saves the email and LLM output.

        Args:
            email (str): The email content.
            llm_output (dict): The LLM output as a dictionary.
        """
        question_id = llm_output["question_id"]
        folder_path = f"{self.email_storage}/{question_id}"
        os.makedirs(folder_path, exist_ok=True)

        # Save email
        email_file_path = f"{folder_path}/mail_{question_id}.txt"
        with open(email_file_path, "w") as file:
            file.write(email)

        # Save LLM output
        llm_output_file_path = f"{folder_path}/output_{question_id}.json"
        with open(llm_output_file_path, "w") as file:
            json.dump(llm_output, file)

    def create_email(self, llm_output: dict) -> str:
        """
        Creates the email.

        Args:
            llm_output (dict): The LLM output as a dictionary.

        Returns:
            The formatted email content as a string.
        """
        email_content = self.email_template.format(
            type_question=llm_output["question_type"],
            llm_output=pformat(llm_output, sort_dicts=False),
        )
        return email_content

    def ask_user(self) -> bool:
        """
        Asks the user if the question should be redirected to an expert.

        Returns:
            True if the user wants to redirect the question to an expert, False otherwise.
        """
        user_reply = input(
            ">>> Ich konnte Ihre Frage nicht beantworten. Frage weiterleiten? (Ja/Nein):\n"
        )
        return user_reply.lower() == "ja"

    def ask_and_foward(self, llm_output: dict) -> None:
        """
        Asks the user and forwards the question to an expert if needed.

        Args:
            llm_output (dict): The LLM output as a dictionary.
        """
        if self.ask_user():
            email = self.create_email(llm_output)
            self.send_email_and_save_json(email, llm_output)
            print(">>> Ihre Frage wurde an einen Experten weitergeleitet.\n")
            logger.info("QUESTION NOT SOLVED! EMAIL SENT TO EXPERT.")

    def get_expert_response(self, txt_file: str) -> tuple:
        """
        Retrieves the expert response from a text file.

        Args:
            txt_file (str): The path to the text file.

        Returns:
            A tuple containing the question ID and the expert's response.
        """
        with open(txt_file, "r") as file:
            contents = file.read()

        human_answer = re.search(
            "<\|human_expert\|>(.*?)<\/\|human_expert\|>", contents, re.DOTALL
        )
        question_id = re.search("'question_id': '([^']+)'", contents, re.DOTALL)

        human_answer = human_answer.group(1).replace("\n", "")
        question_id = question_id.group(1)
        return question_id, human_answer

    def send_expert_response_to_kb(self, txt_file: str) -> None:
        """
        Sends the expert's response to the knowledge base.

        Args:
            txt_file (str): The path to the text file containing the expert's response.
        """
        question_id, human_answer = self.get_expert_response(txt_file)
        doc = self.kb.create_expert_doc(question_id, human_answer, txt_file)
        self.kb.load_doc_to_vector_store(doc)
