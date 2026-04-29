class WorkflowError(Exception):
    """
    Exception indicating an error that the current loop cannot handle, preventing further progress.
    """


class LLMUnavailableError(RuntimeError):
    """
    Raised when the LLM backend fails to respond after all retries.
    Registered as skip_loop_error in QuantRDLoop so a transient LLM outage
    skips the current loop iteration instead of killing the whole process.
    """


class FormatError(WorkflowError):
    """
    After multiple attempts, we are unable to obtain the answer in the correct format to proceed.
    """


class CodeBlockParseError(FormatError):
    """Raised when code block extraction fails after all strategies."""

    def __init__(self, message: str, content: str, language: str) -> None:
        self.message = message
        self.content = content
        self.language = language
        super().__init__(message)


class CoderError(WorkflowError):
    """
    Exceptions raised when Implementing and running code.
    - start: FactorTask => FactorGenerator
    - end: Get dataframe after execution  # nosec

    The more detailed evaluation in dataframe values are managed by the evaluator.  # nosec
    """

    # NOTE: it corresponds to the error of **component**
    caused_by_timeout: bool = False  # whether the error is caused by timeout


class CodeFormatError(CoderError):
    """
    The generated code is not found due format error.
    """


class CustomRuntimeError(CoderError):
    """
    The generated code fail to execute the script.  # nosec
    """


class NoOutputError(CoderError):
    """
    The code fail to generate output file.
    """


class RunnerError(Exception):
    """
    Exceptions raised when running the code output.
    """

    # NOTE: it corresponds to the error of whole **project**


FactorEmptyError = CoderError  # Exceptions raised when no factor is generated correctly

ModelEmptyError = CoderError  # Exceptions raised when no model is generated correctly


class KaggleError(Exception):
    """
    Exceptions raised when calling Kaggle API
    """


class PolicyError(Exception):
    """
    Exceptions raised due to content management policy
    """


class EvaluatorDidNotTerminateError(RuntimeError):
    """
    Evaluator generator did not terminate with a final Feedback.
    """
