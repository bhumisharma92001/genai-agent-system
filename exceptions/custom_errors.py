class GenAIError(Exception):
    pass

class LLMGenerationError(GenAIError):
    pass

class InvalidInputError(GenAIError):
    pass

class InvalidQueryError(GenAIError):
    pass

class ReasoningGenerationError(GenAIError):
    pass

class SummarizationError(GenAIError):
    pass

class EmbeddingError(GenAIError):
    pass

class ModelLoadError(GenAIError):
    pass

class ConfigurationError(GenAIError):
    pass

class IndexingError(GenAIError):
    pass

class ChunkingError(GenAIError):
    pass

class DocumentLoadError(GenAIError):
    pass

class ExtractionError(GenAIError):
    pass

class UnsupportedFormatError(GenAIError):
    pass

class MemoryError(GenAIError):
    pass

class RerankError(GenAIError):
    pass

class RetrievalError(GenAIError):
    pass

class VectorDBError(GenAIError):
    pass

class ToolExecutionError(GenAIError):
    pass

class OrchestrationError(GenAIError):
    pass
