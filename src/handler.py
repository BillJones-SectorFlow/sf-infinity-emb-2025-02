"""
RunPod serverless handler for the Infinity embedding and reranking worker.

This module defines an asynchronous generator function that delegates
requests to the appropriate service methods based on the payload.  It
supports both OpenAI-compatible routes (/v1/models and /v1/embeddings)
and the standard RunPod reranking API.
"""

import runpod
import traceback
import json
import logging
from typing import Any, Dict

from utils import create_error_response
from embedding_service import EmbeddingService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to construct the embedding service at import time so that
# model downloads and engine initialisation occur only once.  Should an
# exception occur during configuration (e.g. missing environment variables),
# fail fast and emit a clear error message.
try:
    embedding_service = EmbeddingService()
except Exception as exc:  # noqa: BLE001  (catch everything to report it cleanly)
    import sys

    sys.stderr.write(f"\nstartup failed: {exc}\n")
    sys.exit(1)


async def async_generator_handler(job: Dict[str, Any]):
    """Handle incoming RunPod jobs.

    The handler inspects the ``input`` portion of the job and chooses
    whether to call an OpenAI-compatible route or a reranking/embedding
    function.  Errors are caught and returned as structured responses.
    """
    logger.info(f"Handler: Received job: {job}")
    
    job_input = job.get("input", {})
    logger.info(f"Handler: Job input keys: {list(job_input.keys())}")
    
    if job_input.get("openai_route"):
        openai_route = job_input.get("openai_route")
        openai_input = job_input.get("openai_input")
        logger.info(f"Handler: OpenAI route: {openai_route}")
        
        if openai_route == "/v1/models":
            call_fn = embedding_service.route_openai_models
            kwargs: Dict[str, Any] = {}
        elif openai_route == "/v1/embeddings":
            # Validate the input
            if not openai_input:
                yield create_error_response("Missing input").model_dump()
                return
            model_name = openai_input.get("model")
            if not model_name:
                yield create_error_response("Did not specify model in openai_input").model_dump()
                return
            call_fn = embedding_service.route_openai_get_embeddings
            kwargs = {
                "embedding_input": openai_input.get("input"),
                "model_name": model_name,
                "return_as_list": True,
            }
        elif openai_route == "/v1/rerank":
            # Validate the input
            if not openai_input:
                yield create_error_response("Missing input").model_dump()
                return
            model_name = openai_input.get("model")
            if not model_name:
                yield create_error_response("Did not specify model in openai_input").model_dump()
                return
            call_fn = embedding_service.infinity_rerank
            kwargs = {
                "query": openai_input.get("query"),
                "docs": openai_input.get("docs"),
                "return_docs": openai_input.get("return_docs"),
                "model_name": model_name,
            }
            logger.info(f"Handler: /v1/rerank kwargs: {kwargs}")
        else:
            yield create_error_response(f"Invalid OpenAI Route: {openai_route}").model_dump()
            return
    else:
        # Standard reranking or embedding request
        if job_input.get("query"):
            call_fn = embedding_service.infinity_rerank
            kwargs = {
                "query": job_input.get("query"),
                "docs": job_input.get("docs"),
                "return_docs": job_input.get("return_docs"),
                "model_name": job_input.get("model"),
            }
        elif job_input.get("input"):
            # Shortcut for embedding-only calls via the standard route
            call_fn = embedding_service.route_openai_get_embeddings
            kwargs = {
                "embedding_input": job_input.get("input"),
                "model_name": job_input.get("model"),
            }
        else:
            yield create_error_response(f"Invalid input: {job}").model_dump()
            return
    
    try:
        logger.info(f"Handler: About to call {call_fn.__name__} with kwargs: {list(kwargs.keys())}")
        out = await call_fn(**kwargs)
        logger.info(f"Handler: Function call successful, response type: {type(out)}")
        logger.info(f"Handler: Response keys: {list(out.keys()) if isinstance(out, dict) else 'NOT_DICT'}")
        
        # Convert response to ensure JSON compatibility
        import json
        try:
            # First test: can we serialize it?
            json_str = json.dumps(out)
            logger.info(f"Handler: Response successfully serialized to JSON ({len(json_str)} chars)")
            
            # Second test: can we round-trip it?
            parsed_back = json.loads(json_str)
            logger.info(f"Handler: Response successfully round-tripped through JSON")
            
            # Return the round-tripped version to ensure clean JSON types
            logger.info(f"Handler: Yielding JSON-clean response...")
            yield parsed_back
            
        except Exception as e:
            logger.error(f"Handler: JSON serialization failed: {e}")
            logger.error(f"Handler: Response content: {out}")
            logger.error(f"Handler: Response type details: {type(out)}")
            
            # Try to identify problematic values
            if isinstance(out, dict):
                for key, value in out.items():
                    try:
                        json.dumps(value)
                    except Exception as ve:
                        logger.error(f"Handler: Key '{key}' has non-serializable value: {value} (type: {type(value)}) - Error: {ve}")
            
            # Return error response
            yield create_error_response(f"Response serialization failed: {e}").model_dump()
        
    except Exception as e:
        # Print the full traceback for debugging
        logger.error(f"Handler: Exception occurred: {e}")
        logger.error(f"Handler: Exception type: {type(e)}")
        traceback.print_exc()
        error_response = create_error_response(str(e)).model_dump()
        logger.error(f"Handler: Yielding error response: {error_response}")
        yield error_response

# When executed as a script, start the RunPod serverless handler.  The
# concurrency modifier ensures we respect the configured maximum concurrency.
# Setting ``return_aggregate_stream`` to ``True`` aggregates all yielded
# outputs from the asynchronous generator into a single response.  Without this
# flag, generator outputs are only available via the ``/stream`` endpoint; by
# enabling it, the aggregated response becomes accessible through the
# synchronous ``/runsync`` endpoint as well【535206775436489†L274-L280】.  This is
# important for rerank requests because the RunPod queue infrastructure
# expects a full JSON payload rather than a token-by-token stream.
if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": async_generator_handler,
            # Limit concurrency per configuration to avoid overloading the GPU.
            "concurrency_modifier": lambda current: embedding_service.config.runpod_max_concurrency,
            # Aggregate streaming outputs so that full results are available via the
            # `/run` and `/runsync` endpoints.
            "return_aggregate_stream": True,
        }
    )



