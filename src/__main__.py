from application.chatbot import ChatBot
from application.knowledge_base import KnowledgeBase
from application.chains import DocumentChain, ProductChain, Judge
from application.models import setup_models


def main():
    """Main function to setup the application and start the chat."""
    embedding_model, llm = setup_models()

    path_kb = "/path/"
    path_sql_db = path_kb + "/sqlite_db.db"
    path_vector_store = path_kb + "/chroma_db"
    path_email_storage = path_kb + "/email_storage"

    kb = KnowledgeBase(
        path_sql_db=path_sql_db,
        path_vector_store=path_vector_store,
        path_email_storage=path_email_storage,
        embedding_model=embedding_model,
    )

    doc_chain = DocumentChain(retriever=kb.retriever, llm=llm)
    product_chain = ProductChain(llm=llm)
    judge = Judge(llm=llm)
    bot = ChatBot(
        llm=llm,
        knowledge_base=kb,
        document_chain=doc_chain,
        product_chain=product_chain,
        judge=judge,
    )

    return bot


if __name__ == "__main__":
    bot = main()
    chat_history = bot.start_chat()
