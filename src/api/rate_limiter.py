"""Module with rate limiter for FastAPI endpoints."""

from datetime import UTC, datetime, timedelta

from fastapi import HTTPException
from loguru import logger

from src.configuration import config


class RateLimiter:
    """In-memory rate limiter performing checks ."""

    def __init__(
        self,
        max_request_per_interval: int = config.api_max_requests_per_interval,
        interval: timedelta = config.api_rate_limiter_interval,
    ) -> None:
        """
        Set up parameters and an in-memory storage.

        Args:
            max_request_per_interval (int, optional): The maximum number
                of request per selected interval of time.
                Defaults to the value from the configurartion.
            interval (timedelta, optional): The interval of time, for which a limit of
                request is set per user. Defaults to the value from the configuration.

        Raises:
            ValueError: Raised if `max_request_per_interval` is lower than 1
                (request per interval).
        """
        if max_request_per_interval < 1:
            raise ValueError("`max_request_per_interval` must be >= 1.")

        recommended_minimum_interval = timedelta(seconds=1)
        if interval < recommended_minimum_interval:
            logger.warning(
                "From the performance perspective at least one second is recommended "
                f"for  {RateLimiter.__name__}'s `interval`."
            )
        self._interval = interval

        self._max_requests_per_interval = max_request_per_interval
        self._requests: dict[str, set[datetime]] = {}

    def __call__(self, identifier: str) -> None:
        """
        Report a new request and check if it abuses the limit.

        Args:
            identifier (str): Identifier, such as: IP address, API key or other unique
                information differentiating users.

        Raises:
            HTTPException: Raised if the limit has been exceeded by a user.
        """
        now = datetime.now(tz=UTC)
        # Accept the first call ever.
        if identifier not in self._requests:
            self._requests[identifier] = {now}
            return

        # Removed the expired requests.
        self._requests[identifier] = {
            request
            for request in self._requests[identifier]
            if request + self._interval > now
        }

        if len(self._requests[identifier]) >= self._max_requests_per_interval:
            self._requests[identifier].add(now)
            raise HTTPException(
                status_code=429,
                detail=(
                    f"You are allowed to send {self._max_requests_per_interval} "
                    f"request(s) every {self._interval}. You have been blocked because "
                    "of exceeding this limit. Please, try again later!"
                ),
            )

        self._requests[identifier].add(now)
