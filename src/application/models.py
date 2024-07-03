from typing import Tuple

import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from llama_cpp import LlamaGrammar
from utils.logging_utils import logger


def setup_models() -> Tuple[HuggingFaceEmbeddings, LlamaCpp]:
    """
    Sets up the models for the application.

    Returns:
        A tuple containing the initialized HuggingFaceEmbeddings and LlamaCpp models.
    """
    embedding_model = setup_embeddings()
    llm = setup_llm()

    return embedding_model, llm


def setup_llm() -> LlamaCpp:
    """
    Sets up the LlamaCpp model.

    Returns:
        The initialized LlamaCpp model.
    """
    None if torch.cuda.is_available() else logger.warning("CUDA is not enabled".upper())

    gpu_cpu_config = {
        "n_ctx": 3900,
        "n_gpu_layers": 31,
        # "n_gpu_layers": 15,
        "n_threads": 12,
    }

    grammer_path = "/path/json_grammer.gbnf"
    model_path = "/path/sauerkrautlm-7b-hero.Q5_K_M.gguf"

    llm_grammar = LlamaGrammar.from_file(grammer_path, verbose=False)

    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.5,
        max_tokens=2048,
        repeat_penalty=1.1,
        top_p=1,
        verbose=False,
        grammar=llm_grammar,
        stop=["\n\n"],
        f16_kv=True,
        **gpu_cpu_config,
    )

    return llm


def setup_embeddings() -> HuggingFaceEmbeddings:
    """
    Sets up the HuggingFaceEmbeddings model.

    Returns:
        The initialized HuggingFaceEmbeddings model.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        # model_kwargs={"device": "cuda"},
    )

    return embedding_model
