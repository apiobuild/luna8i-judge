class PromptConsistencyError(ValueError):
    """Raised when a row's user message doesn't match the prompt_template regex."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"{reason}.")
        self.reason = reason
