class GenAIError(Exception):
    """Base class for all system errors. Baaki saare errors isse inherit karenge."""
    pass

class ReasoningGenerationError(GenAIError):
    """LLM generation failure ke liye."""
    pass

class InvalidQueryError(GenAIError):
    """Empty or malformed user query ke liye."""
    pass

class InvalidInputError(GenAIError):
    """Empty or invalid input."""
    pass

class SummarizationError(GenAIError):
    """Summarization failure ke liye."""
    pass

class EmbeddingError(GenAIError):
    """Embedding generation failure."""
    pass

class ModelLoadError(GenAIError):
    """Model loading failure."""
    pass

class ConfigurationError(GenAIError):
    """System configuration ya .env variables missing/invalid hone par."""
    pass

class IndexingError(GenAIError):
    """Document indexing failure."""
    pass

class ChunkingError(GenAIError):
    """Chunk creation failure."""
    pass

class DocumentLoadError(GenAIError):
    """Document loading failure."""
    pass

class ExtractionError(GenAIError):
    """Table or text extraction failure."""
    pass

class UnsupportedFormatError(GenAIError):
    """Unsupported file format."""
    pass

class LLMGenerationError(GenAIError):
    """LLM API call failure."""
    pass

class MemoryError(GenAIError):
    """Memory read/write failure."""
    pass

class RerankError(GenAIError):
    """Reranking failure."""
    pass

class RetrievalError(GenAIError):
    """Retrieval failure."""
    pass

class VectorDBError(GenAIError):
    """Vector DB operation failure."""
    pass