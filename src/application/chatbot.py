import application.templates as tl
from langchain_community.llms import LlamaCpp
from application.chains import Judge, ProductChain, DocumentChain
from application.knowledge_base import KnowledgeBase
from utils.logging_utils import logger
from application.communcation_handler import CommunicationHandler
import uuid


class ChatBot:
    """Chatbot class that orchestrates the conversation between the user and the application components."""

    def __init__(
        self,
        llm: LlamaCpp,
        knowledge_base: KnowledgeBase,
        document_chain: DocumentChain,
        product_chain: ProductChain,
        judge: Judge,
    ):
        """
        Initializes a ChatBot instance.

        Args:
            llm (LlamaCpp): The LlamaCpp instance for language processing.
            knowledge_base (KnowledgeBase): The KnowledgeBase instance for accessing the knowledge base.
            document_chain (DocumentChain): The DocumentChain instance for document retrieval.
            product_chain (ProductChain): The ProductChain instance for product-related queries.
            judge (Judge): The Judge instance for evaluating the LLM output.
        """
        self.llm = llm
        self.kb = knowledge_base
        self.document_chain = document_chain
        self.product_chain = product_chain
        self.judge = judge
        self.comm_handler = CommunicationHandler(knowledge_base=self.kb)
        self.chat_history = []
        self.question_id = 0

    def say_message(self, hello_message: bool):
        """
        Prints a message to the console.

        Args:
            hello_message (bool): Indicates whether it is a hello message or not.
        """
        print(
            "---------------------------------------------------------------------------------"
        )
        if hello_message:
            print(tl.greeting_template)
        else:
            print("Auf Wiedersehen! 🦜")
        print(
            "---------------------------------------------------------------------------------"
        )

    def get_chat_history(self) -> list:
        """Returns the chat history."""
        return self.chat_history

    def append_to_chat_history(self, message: str):
        """Appends a message to the chat history."""
        self.chat_history.append(message)

    def call_judge(self, llm_output: str):
        """Calls the Judge instance to evaluate the LLM output.

        Args:
            llm_output (str): The LLM output to evaluate.
        """
        self.judge.label = "chat"
        judge_response = self.judge.execute(llm_response=llm_output)
        if judge_response is not None:
            self.comm_handler.ask_and_foward(judge_response)

    def call_doc_chain(self, init_query: str):
        """
        Calls the DocumentChain instance to execute a document retrieval.

        Args:
            init_query (str): The initial query for document retrieval.
        """
        self.document_chain.question_id = uuid.uuid4().hex
        llm_output = self.document_chain.execute(query=init_query)
        self.append_to_chat_history(llm_output)
        self.call_judge(llm_output)

    def call_product_chain(self, product_info: dict):
        """
        Calls the ProductChain instance to execute product-related queries.

        Args:
            product_info (dict): The product information.
        """
        while True:
            product_query = input(">>> Was möchten Sie über das Produkt wissen?\n")
            self.product_chain.question_id = uuid.uuid4().hex
            llm_output = self.product_chain.execute(
                query=product_query,
                product_info=product_info,
            )
            self.append_to_chat_history(llm_output)
            self.call_judge(llm_output)
            product_query = input(
                ">>> Haben Sie eine weitere Frage zum Produkt? (Ja/Nein)\n"
            )
            if product_query.lower() == "nein":
                break

    def start_chat(self) -> list:
        """
        Starts the chatbot and handles the conversation.

        Returns:
            list: The chat history.
        """
        logger.info("###### NEW CHAT ######.")
        self.say_message(hello_message=True)
        while True:
            init_query = input("\n>>> Ihre Frage:\n")
            logger.info(f"USER INPUT: {init_query}")
            if init_query == "exit" or init_query == "quit":
                self.say_message(hello_message=False)
                logger.info("###### END CHAT ######.")
                break
            if init_query == "":
                continue
            if init_query.isdigit():
                product_info = self.kb.execute_sql_query(product_code=init_query)
                self.call_product_chain(product_info)
            else:
                self.call_doc_chain(init_query)

        return self.get_chat_history()
