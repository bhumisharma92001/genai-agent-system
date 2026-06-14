from exceptions.custom_errors import ConfigurationError


class LLMConfig:

    def __init__(
        self,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 512
    ):
        if not 0 <= temperature <= 2.0:
            raise ConfigurationError("temperature must be between 0 and 2.0")
        if not 0 <= top_p <= 1:
            raise ConfigurationError("top_p must be between 0 and 1")
        if max_tokens <= 0:
            raise ConfigurationError("max_tokens must be positive")

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens