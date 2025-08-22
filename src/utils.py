"""
Utility functions and Pydantic models for Infinity embedding and reranking.

This module defines the schema for OpenAI-compatible responses as well as
helpers to convert raw data into those schemas.  It also provides a
consistent error response format for returning validation errors to the
caller.
"""

from http import HTTPStatus
from typing import Any, Dict, Iterable, List, Optional, Union
from uuid import uuid4
import time
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, conlist


# When importing Pydantic 2, some types may not be available; fall back to
# Pydantic 1 where appropriate.  ``StringConstraints`` is only present in
# Pydantic 2, so we attempt to import it and fall back to ``constr``
try:
    from pydantic import StringConstraints  # type: ignore

    INPUT_STRING = StringConstraints(max_length=8192 * 15, strip_whitespace=True)
    ITEMS_LIMIT = {
        "min_length": 1,
        "max_length": 8192,
    }
except ImportError:
    from pydantic import constr  # type: ignore

    INPUT_STRING = constr(max_length=8192 * 15, strip_whitespace=True)  # type: ignore
    ITEMS_LIMIT = {
        "min_items": 1,
        "max_items": 8192,
    }


# Type alias describing the ndarray returned from Infinity.  The values are
# typically float32 but may vary depending on the dtype selected.
EmbeddingReturnType = npt.NDArray[Union[np.float32, np.float32]]  # type: ignore


class ErrorResponse(BaseModel):
    """Standardised error response for failed requests."""

    object: str = "error"
    message: str
    type: str
    param: Optional[str] = None
    code: int


def create_error_response(
    message: str,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> ErrorResponse:
    """Helper to construct an error response.

    Args:
        message: Human-readable error message.
        err_type: Identifier for the error type.
        status_code: HTTP status to return.

    Returns:
        An ``ErrorResponse`` instance with the provided fields filled in.
    """
    return ErrorResponse(message=message, type=err_type, code=status_code.value)


class _EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class ModelInfo(BaseModel):
    id: str
    stats: Dict[str, Any]
    object: str = "model"
    owned_by: str = "infinity"
    created: int = int(time.time())
    backend: str = ""


class OpenAIModelInfo(BaseModel):
    data: List[ModelInfo] = Field(default_factory=list)
    object: str = "list"


class OpenAIEmbeddingResult(BaseModel):
    object: str = "list"
    data: List[_EmbeddingObject]
    model: str
    usage: _Usage


def list_embeddings_to_response(
    embeddings: Union[EmbeddingReturnType, Iterable[EmbeddingReturnType]],
    model: str,
    usage: int,
) -> Dict[str, Any]:
    """Convert raw embedding array into an OpenAI-compatible response.

    Args:
        embeddings: A numpy array of embeddings or an iterable of arrays.
        model: Model name used to compute embeddings.
        usage: Total number of tokens processed.

    Returns:
        A dictionary conforming to the OpenAI embedding response schema.
    """
    return {
        "model": model,
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": emb.tolist(),
                "index": idx,
            }
            for idx, emb in enumerate(embeddings)
        ],
        "usage": {"prompt_tokens": usage, "total_tokens": usage},
    }


def to_rerank_response(
    scores: List[float],
    model: str,
    usage: int,
    documents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a response object for reranking requests.

    Args:
        scores: List of relevance scores.
        model: Model name used to compute scores.
        usage: Total number of tokens processed.
        documents: Optional list of documents.  If provided, they will be
            attached to each result; otherwise only scores and indices are
            returned.

    Returns:
        A dictionary conforming to the expected reranking response format.
    """
    if documents is None:
        return {
            "model": model,
            "results": [
                {"relevance_score": score, "index": idx}
                for idx, score in enumerate(scores)
            ],
            "usage": {"prompt_tokens": usage, "total_tokens": usage},
        }
    else:
        return {
            "model": model,
            "results": [
                {"relevance_score": score, "index": idx, "document": doc}
                for idx, (score, doc) in enumerate(zip(scores, documents))
            ],
            "usage": {"prompt_tokens": usage, "total_tokens": usage},
        }
