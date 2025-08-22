"""
Configuration utilities for the embedding and reranking worker.

This module encapsulates how environment variables are parsed and converted
into Python-native values.  The defaults mirror the upstream Infinity
worker but expose additional flexibility through environment variables.  All
environment variables are optional except for `MODEL_NAMES`, which must be
provided at runtime.
"""

import os
from functools import cached_property
from typing import List
from dotenv import load_dotenv


# Default values for various configuration options.  These values are only
# applied when the corresponding environment variable is not set.
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_BACKEND: str = "torch"


# Ensure the queue size has a finite upper bound.  Without this, the
# Infinity engine may attempt to allocate unbounded memory for the request
# queue, potentially leading to OOM conditions.  Only set a default when
# not provided by the user.
if not os.environ.get("INFINITY_QUEUE_SIZE"):
    os.environ["INFINITY_QUEUE_SIZE"] = "48000"


class EmbeddingServiceConfig:
    """Parse runtime configuration from environment variables.

    An instance of this class is attached to the embedding service and
    referenced by the RunPod handler.  Values are computed lazily via
    ``cached_property`` so that they are only parsed once per process.
    """

    def __init__(self) -> None:
        # Load variables from a .env file if present.  This makes it easier
        # to run the worker locally without exporting dozens of environment
        # variables.
        load_dotenv()

    def _get_no_required_multi(self, name: str, default=None) -> List[str]:
        """Return a list of values for a semicolon separated environment variable.

        The Infinity CLI supports broadcasting single values across multiple
        models by repeating the default value.  This helper mirrors that
        behaviour and raises a ValueError when the number of provided values
        does not match the number of configured model names.

        Args:
            name: Name of the environment variable to read.
            default: Default value to broadcast if the variable is unset.

        Returns:
            A list of strings, one for each model name.
        """
        # Broadcast the default when the variable is unset.  Note that we
        # generate one extra ``;`` so ``split`` preserves empty values at
        # the end; these empty values are then filtered out below.
        values = os.getenv(name, f"{default};" * len(self.model_names)).split(";")
        values = [v for v in values if v]
        if len(values) != len(self.model_names):
            raise ValueError(
                f"Env var: {name} must have the same number of elements as MODEL_NAMES"
            )
        return values

    @cached_property
    def backend(self) -> str:
        """Return the inference backend to use (e.g. 'torch', 'optimum', 'ctranslate2')."""
        return os.environ.get("BACKEND", DEFAULT_BACKEND)

    @cached_property
    def model_names(self) -> List[str]:
        """Return the list of HuggingFace model names configured for the worker.

        An exception is raised when ``MODEL_NAMES`` is unset or empty because
        Infinity requires at least one model to be provided.
        """
        model_names_env = os.environ.get("MODEL_NAMES")
        if not model_names_env:
            raise ValueError(
                "Missing required environment variable 'MODEL_NAMES'.\n"
                "Please provide at least one HuggingFace model ID, or multiple IDs separated by a semicolon.\n"
                "Examples:\n"
                "  MODEL_NAMES=BAAI/bge-small-en-v1.5\n"
                "  MODEL_NAMES=BAAI/bge-small-en-v1.5;intfloat/e5-large-v2\n"
            )
        names = [mn for mn in model_names_env.split(";") if mn]
        return names

    @cached_property
    def batch_sizes(self) -> List[int]:
        """Return the per-model batch sizes to use during inference."""
        sizes = self._get_no_required_multi("BATCH_SIZES", DEFAULT_BATCH_SIZE)
        return [int(size) for size in sizes]

    @cached_property
    def dtypes(self) -> List[str]:
        """Return the per-model dtype specifications (e.g. 'auto', 'fp16', 'fp8')."""
        return self._get_no_required_multi("DTYPES", "auto")

    @cached_property
    def runpod_max_concurrency(self) -> int:
        """Return the maximum number of concurrent jobs accepted by RunPod.

        RunPod will call the handler with this concurrency limit.  Use a
        conservative default of 10 to avoid GPU memory exhaustion.
        """
        return int(os.environ.get("RUNPOD_MAX_CONCURRENCY", 10))
