from langchain_ollama import OllamaEmbeddings


def create_embedding_model() -> OllamaEmbeddings:
    """
    Creates and returns an instance of OllamaEmbeddings using the "nomic-embed-text" model.

    :return: Configured instance of OllamaEmbeddings.
    """
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    return embedding_model
