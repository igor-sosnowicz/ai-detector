"""Package with data models for the API."""

from enum import Enum

from pydantic import BaseModel


class HealthcheckResponse(BaseModel):
    """Response from the healthcheck endpoint indicating the status of the system."""

    is_healthy: bool


class EvaluationMethod(str, Enum):
    """Valid methods of text evaluation varying by inference time and accuracy."""

    MOST_ACCURATE = "most_accurate"
    MORE_ACCURATE = "more_accurate"
    BALANCED = "balanced"
    FAST = "fast"
    FASTEST = "fastest"


class LLMDetectionRequest(BaseModel):
    """API request for a text origin evaluation."""

    text: str
    method: EvaluationMethod = EvaluationMethod.MOST_ACCURATE


class LLMDetectionResponse(BaseModel):
    """Response sent when a client requests detection in a text."""

    is_llm_generated: bool
    confidence: float
    remarks: str
