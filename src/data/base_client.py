"""Base HTTP client with rate limiting and retry logic."""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self.last_request_time = time.time()


class APIError(Exception):
    """Base exception for API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    pass


class BaseClient(ABC):
    """Abstract base class for API clients."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        requests_per_minute: int = 60,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self) -> None:
        """Configure session with default headers."""
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "algo-quant/0.1.0",
            }
        )

    @abstractmethod
    def _get_auth_params(self) -> dict[str, str]:
        """Return authentication parameters for the API."""
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.RequestException, RateLimitError)),
    )
    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with rate limiting and retry logic."""
        self.rate_limiter.wait()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        params = params or {}
        params.update(self._get_auth_params())

        logger.debug(f"{method.upper()} {url}")

        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                timeout=self.timeout,
                **kwargs,
            )

            if response.status_code == 401:
                raise AuthenticationError("Invalid API key", status_code=401)
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded", status_code=429)
            elif response.status_code >= 400:
                raise APIError(
                    f"API error: {response.text}", status_code=response.status_code
                )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.error(f"Request timeout: {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get(
        self, endpoint: str, params: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Make a GET request."""
        return self._request("GET", endpoint, params=params, **kwargs)

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self) -> "BaseClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
