import sqlite3
from typing import Any, Dict, List

from chromadb import PersistentClient
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_core.documents.base import Document
import json
from datetime import datetime


class KnowledgeBase:
    """Represents a knowledge base for storing and retrieving documents."""

    def __init__(
        self,
        path_sql_db: str,
        path_vector_store: str,
        path_email_storage: str,
        embedding_model: HuggingFaceEmbeddings,
    ):
        """
        Initializes a KnowledgeBase object.

        Args:
            path_sql_db (str): The path to the SQLite database file.
            path_vector_store (str): The path to the vector store file.
            path_email_storage (str): The path to the email storage directory.
            embedding_model (HuggingFaceEmbeddings): The embedding model used for vectorization.
        """
        self.path_sql_db = path_sql_db
        self.path_vector_store = path_vector_store
        self.path_email_storage = path_email_storage
        self.embedding_model = embedding_model
        self.sql_db = self.setup_sql_database()
        self.vector_store = self.setup_vector_store()
        self.retriever = self.create_retriever()

    def display_vector_store_info(self) -> None:
        """Displays information about the vector store."""
        collections = self.vector_store._client.list_collections()
        print("Store settings:")

        for collection in collections:
            example = collection.peek(limit=1)
            print(f"Collection {collection.name} with ID {collection.id}")
            print(f"Total number of embeddings: {collection.count()}")
            print(f"Example {example['documents']}\n")

    def display_sql_db_info(self) -> None:
        """Displays information about the SQL database."""
        table_context = self.sql_db.get_context()
        print("SQL database settings:")
        print(f"Contains following tables: {(self.sql_db.get_usable_table_names())}")
        print(f"Table context: {table_context['table_info']}")

    def setup_sql_database(self) -> SQLDatabase:
        """
        Sets up the SQL database.

        Returns:
            SQLDatabase: The SQL database object.
        """
        return SQLDatabase.from_uri("sqlite:///" + self.path_sql_db)

    def setup_vector_store(self) -> Chroma:
        """
        Sets up the vector store.

        Returns:
            Chroma: The vector store object.
        """
        return Chroma(
            client=PersistentClient(self.path_vector_store),
            embedding_function=self.embedding_model,
            collection_name="technical_documents",
        )

    def execute_sql_query(self, product_code: int) -> List[Dict[str, Any]]:
        """
        Executes an SQL query and returns the results.

        Args:
            product_code (int): The product code to search for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the rows returned by the query.
        """
        query = f"SELECT * FROM lamps WHERE Bestell_nr = {int(product_code)};"

        conn = sqlite3.connect(self.path_sql_db)
        c = conn.cursor()
        c.execute(query)
        rows = c.fetchall()
        column_names = [description[0] for description in c.description]
        conn.close()

        # Return a list of dictionaries, each representing a row
        return [dict(zip(column_names, row)) for row in rows][0]

    def create_retriever(self) -> None:
        """Creates a retriever for the vector store."""
        return self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        )

    def load_doc_to_vector_store(self, doc: Document) -> None:
        """
        Loads a document into the vector store.

        Args:
            doc (Document): The document to load.
        """
        self.vector_store.add_documents([doc])

    def get_docs(self, keywords: str = None) -> List[Document]:
        """
        Retrieves documents from the vector store.

        Args:
            keywords (str, optional): Keywords to filter the documents. Defaults to None.

        Returns:
            List[Document]: A list of documents.
        """
        if keywords:
            return self.vector_store.get(where={"keywords": keywords})
        else:
            return self.vector_store.get()

    def create_expert_doc(
        self, question_id: int, expert_answer: str, source: str
    ) -> Document:
        """
        Creates from an human expert answer a LangChain Document.

        Args:
            question_id (int): The ID of the question.
            expert_answer (str): The expert's answer.
            source (str): The source of the answer.

        Returns:
            Document: The created document.
        """
        path = f"{self.path_email_storage}/output_{question_id}.json"
        with open(path, "r") as file:
            json_file = json.load(file)

        question = json_file["question"]
        page_content = f"Frage: {question}. Antwort: {expert_answer}"
        doc = Document(
            page_content=page_content,
            metadata={
                "question_id": question_id,
                "source": source,
                "author": "Expert",
                "creationDate": datetime.now().isoformat(),
                "keywords": "expert_answer",
            },
        )
        return doc
