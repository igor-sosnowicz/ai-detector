"""Module with the API utility functions."""

from fastapi import HTTPException, Request


def get_ip_address_or_raise(fastapi_request: Request) -> str:
    """
    Get an IP address of the client sending the request raising an exception if missing.

    Args:
        fastapi_request (Request): The request of the client.

    Raises:
        HTTPException: Raised if an IP address cannot be retrieved from the request.

    Returns:
        str: IP address of the client.
    """
    ip = fastapi_request.headers.get("X-Forwarded-For")
    if ip:
        return ip
    if fastapi_request.client is None:
        raise HTTPException(
            detail=(
                "Unable to identify the IP address. Please, do not use proxy while "
                "connecting to this API."
            ),
            status_code=401,
        )
    return fastapi_request.client.host


def get_length_adjusted_confidence(
    confidence: float, text: str
) -> tuple[float, str | None]:
    """
    Get confidence of a text belonging to a class adjusted by its length.

    Args:
        confidence (float): Original confidence of text belonging to a class.
        text (str): The text, for which a confidence was provided.

    Returns:
        tuple[float, str | None]: Length-adjusted confidence. The shorter the text,
            the higher the length penalty on confidence will be.
    """
    words = len(text.split())
    min_words = 200
    if words > min_words:
        return (confidence, None)

    remark = (
        "The provided text is very short. The accuracy of LLM-written "
        "detection is better on longer texts. Confidence of the model "
        "has been reduced to reflect that. "
        f"Please, provide a text longer than {words} words. "
    )

    adjusted_confidence = confidence - (min_words - words) / min_words
    return (adjusted_confidence, remark)


def map_score_to_confidence(score: float) -> float:
    """
    Map score to confidence based on the range.

    If score >= 0.5: map [0.5, 1.0] to [0.0, 1.0]
    If score < 0.5: map [0.0, 0.5] to [0.0, 1.0]

    Args:
        score (float): The raw score in range [0.0, 1.0]

    Returns:
        float: Confidence value in range [0.0, 1.0]
    """
    half = 0.5
    if score >= half:
        # Map [0.5, 1.0] to [0.0, 1.0].
        return (score - 0.5) * 2

    # Map [0.0, 0.5] to [0.0, 1.0].
    return score * 2
