"""
High-level service encapsulating Infinity engines and reranking/classifier models.

This class manages multiple Infinity engine instances for embedding and
reranking tasks.  It also wraps MXBAI classifier models for reranking
that require classification rather than similarity scoring.  Environment
variables control which models are loaded and how the engines are
configured.  See ``config.EmbeddingServiceConfig`` for details.
"""

import asyncio
import os
import logging
from typing import Dict, List, Union, Optional, Any

from infinity_emb.engine import AsyncEngineArray, EngineArgs

from config import EmbeddingServiceConfig
from utils import (
    OpenAIModelInfo,
    ModelInfo,
    list_embeddings_to_response,
    to_rerank_response,
)
from classifier_service import MXBAIRerankClassifier

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service providing embedding, reranking and classification functionality."""

    def __init__(self) -> None:
        self.config = EmbeddingServiceConfig()
        logger.info(f"EmbeddingService: Initializing with config model_names: {self.config.model_names}")
        
        # Build engine arguments for each model configured.  Additional
        # parameters such as revision, trust_remote_code and bettertransformer
        # can be supplied via environment variables; see below.
        engine_args: List[EngineArgs] = []
        # Prepare a dictionary of extra parameters recognised by EngineArgs.
        extra_args: Dict[str, Any] = {}
        # HuggingFace revision to load (e.g. a branch or commit SHA)
        rev = os.environ.get("REVISION") or os.environ.get("INFINITY_REVISION")
        if rev:
            extra_args["revision"] = rev
        # Whether to trust remote code when downloading models
        trust_remote = os.environ.get("TRUST_REMOTE_CODE") or os.environ.get("INFINITY_TRUST_REMOTE_CODE")
        if trust_remote is not None:
            # Any value other than explicit false/0/no is treated as True
            extra_args["trust_remote_code"] = str(trust_remote).lower() not in ["0", "false", "no"]
        # Control the BetterTransformer optimisation.  Prefer explicit
        # NO_BETTERTRANSFORMER over BETTERTRANSFORMER if both are set.
        no_bt = os.environ.get("NO_BETTERTRANSFORMER")
        if no_bt is not None:
            extra_args["bettertransformer"] = False
        else:
            bt = os.environ.get("BETTERTRANSFORMER") or os.environ.get("INFINITY_BETTERTRANSFORMER")
            if bt is not None:
                extra_args["bettertransformer"] = str(bt).lower() not in ["0", "false", "no"]
        # Embedding dtype after the forward pass (e.g. 'float32', 'int8')
        emb_dtype = os.environ.get("EMBEDDING_DTYPE") or os.environ.get("INFINITY_EMBEDDING_DTYPE")
        if emb_dtype:
            extra_args["embedding_dtype"] = emb_dtype
        # Pooling method to use for embeddings; Infinity will infer this if unset
        pooling = os.environ.get("POOLING_METHOD") or os.environ.get("INFINITY_POOLING_METHOD")
        if pooling:
            extra_args["pooling_method"] = pooling

        # Compose EngineArgs for each configured model
        for model_name, batch_size, dtype in zip(
            self.config.model_names, self.config.batch_sizes, self.config.dtypes
        ):
            logger.info(f"EmbeddingService: Creating engine args for model: '{model_name}', batch_size: {batch_size}, dtype: {dtype}")
            args = {
                "model_name_or_path": model_name,
                "batch_size": batch_size,
                "engine": self.config.backend,
                "dtype": dtype,
                "model_warmup": False,
                # Use real token counts for length calculation
                "lengths_via_tokenize": True,
            }
            args.update(extra_args)
            engine_args.append(EngineArgs(**args))

        # Create a single AsyncEngineArray for all models.  Engines are loaded
        # lazily in the ``start`` method to allow async initialisation.
        self.engine_array = AsyncEngineArray.from_args(engine_args)
        self.is_running: bool = False
        # Semaphore controlling concurrent engine usage.  Limit concurrency to
        # whatever the RunPod handler is configured to allow.  Without this
        # concurrency control, simultaneous requests could exhaust GPU memory.
        self.semaphore = asyncio.Semaphore(self.config.runpod_max_concurrency)
        # Optional classifier services keyed by model name
        self.classifier_services: Dict[str, MXBAIRerankClassifier] = {}
        
        logger.info(f"EmbeddingService: Checking models for classifier initialization...")
        for model in self.config.model_names:
            logger.info(f"EmbeddingService: Evaluating model '{model}' for classifier initialization")
            logger.info(f"EmbeddingService: Model '{model}' contains 'rerank-large-v2-seq': {'rerank-large-v2-seq' in model}")
            logger.info(f"EmbeddingService: Model '{model}' contains 'rerank-base-v2-seq': {'rerank-base-v2-seq' in model}")
            
            if "rerank-large-v2-seq" in model or "rerank-base-v2-seq" in model:
                logger.info(f"EmbeddingService: Model '{model}' matches classifier criteria, attempting initialization...")
                try:
                    self.classifier_services[model] = MXBAIRerankClassifier(model)
                    print(f"Initialized classifier for {model}")
                    logger.info(f"EmbeddingService: Successfully initialized classifier for model '{model}'")
                except Exception as exc:
                    print(f"Warning: Could not initialize classifier for {model}: {exc}")
                    logger.error(f"EmbeddingService: Failed to initialize classifier for model '{model}': {exc}")
            else:
                logger.info(f"EmbeddingService: Model '{model}' does not match classifier criteria, skipping")
        
        logger.info(f"EmbeddingService: Final classifier_services keys: {list(self.classifier_services.keys())}")
        logger.info(f"EmbeddingService: Total classifiers initialized: {len(self.classifier_services)}")

    async def start(self) -> None:
        """Start all Infinity engines asynchronously on first use."""
        async with self.semaphore:
            if not self.is_running:
                logger.info("EmbeddingService: Starting engine array...")
                await self.engine_array.astart()
                self.is_running = True
                logger.info("EmbeddingService: Engine array started successfully")

    async def stop(self) -> None:
        """Stop all Infinity engines asynchronously."""
        async with self.semaphore:
            if self.is_running:
                logger.info("EmbeddingService: Stopping engine array...")
                await self.engine_array.astop()
                self.is_running = False
                logger.info("EmbeddingService: Engine array stopped successfully")

    async def route_openai_models(self) -> Dict[str, Any]:
        """List all configured models in an OpenAI-compatible format."""
        models = self.list_models()
        logger.info(f"EmbeddingService: route_openai_models returning {len(models)} models: {models}")
        return OpenAIModelInfo(
            data=[ModelInfo(id=mid, stats={}) for mid in models]
        ).model_dump()

    def list_models(self) -> List[str]:
        """Return a list of served model identifiers."""
        models = list(self.engine_array.engines_dict.keys())
        logger.info(f"EmbeddingService: list_models returning: {models}")
        return models

    async def route_openai_get_embeddings(
        self,
        embedding_input: Union[str, List[str]],
        model_name: str,
        return_as_list: bool = False,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Return embeddings for a single string or a list of strings.

        Args:
            embedding_input: Text or list of texts to embed.
            model_name: Which model to use.
            return_as_list: When True, wrap the result in a list to satisfy
                some client expectations (e.g. LiteLLM's aggregator).

        Returns:
            Either a single embedding response or a list of one response.
        """
        logger.info(f"EmbeddingService: route_openai_get_embeddings called with model_name: '{model_name}'")
        
        if not self.is_running:
            await self.start()
        # Normalise the input into a list of strings
        sentences: List[str]
        if isinstance(embedding_input, list):
            sentences = embedding_input
        else:
            sentences = [embedding_input]
        
        logger.info(f"EmbeddingService: Getting embeddings for {len(sentences)} sentences using model '{model_name}'")
        embeddings, usage = await self.engine_array[model_name].embed(sentences)
        response = list_embeddings_to_response(embeddings, model=model_name, usage=usage)
        if return_as_list:
            return [response]
        return response

    async def classifier_rerank(
        self, query: str, docs: List[str], return_docs: bool, model_name: str
    ) -> Dict[str, Any]:
        """Perform reranking via MXBAI classifier instead of Infinity engine."""
        logger.info(f"EmbeddingService: classifier_rerank called with model_name: '{model_name}'")
        
        clf = self.classifier_services.get(model_name)
        if not clf:
            logger.error(f"EmbeddingService: No classifier available for model '{model_name}'")
            logger.error(f"EmbeddingService: Available classifier keys: {list(self.classifier_services.keys())}")
            raise ValueError(f"No classifier available for model {model_name}")
        
        logger.info(f"EmbeddingService: Using classifier for model '{model_name}' to rerank {len(docs)} documents")
        
        try:
            scores, usage = await clf.arerank(query, docs)
            logger.info(f"EmbeddingService: Classifier returned {len(scores)} scores, usage: {usage}")
            logger.info(f"EmbeddingService: Scores: {scores}")
            logger.info(f"EmbeddingService: Scores type: {type(scores)}, individual score types: {[type(s) for s in scores[:3]]}")
            logger.info(f"EmbeddingService: Usage type: {type(usage)}")
            
            # Ensure scores are proper Python floats, not numpy/torch types
            cleaned_scores = [float(score) for score in scores]
            cleaned_usage = int(usage)
            
            logger.info(f"EmbeddingService: Cleaned scores: {cleaned_scores}")
            logger.info(f"EmbeddingService: Cleaned usage: {cleaned_usage}")
            
            docs_to_return: Optional[List[str]] = docs if return_docs else None
            logger.info(f"EmbeddingService: return_docs={return_docs}, docs_to_return type: {type(docs_to_return)}")
            
            logger.info(f"EmbeddingService: Calling to_rerank_response...")
            response = to_rerank_response(scores=cleaned_scores, documents=docs_to_return, model=model_name, usage=cleaned_usage)
            logger.info(f"EmbeddingService: to_rerank_response returned type: {type(response)}")
            logger.info(f"EmbeddingService: Response keys: {list(response.keys()) if isinstance(response, dict) else 'NOT_DICT'}")
            logger.info(f"EmbeddingService: Full response: {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"EmbeddingService: Exception in classifier_rerank: {e}")
            logger.error(f"EmbeddingService: Exception type: {type(e)}")
            import traceback
            logger.error(f"EmbeddingService: Traceback: {traceback.format_exc()}")
            raise

    async def infinity_rerank(
        self, query: str, docs: List[str], return_docs: bool, model_name: str
    ) -> Dict[str, Any]:
        """Perform reranking via Infinity engine or classifier when applicable."""
        logger.info(f"EmbeddingService: infinity_rerank called with model_name: '{model_name}'")
        logger.info(f"EmbeddingService: Current classifier_services keys: {list(self.classifier_services.keys())}")
        logger.info(f"EmbeddingService: Checking if '{model_name}' is in classifier_services...")
        
        # Use the classifier when available for this model
        model_in_classifiers = model_name in self.classifier_services
        logger.info(f"EmbeddingService: model_name '{model_name}' in classifier_services: {model_in_classifiers}")
        
        if model_in_classifiers:
            logger.info(f"EmbeddingService: Using classifier for model '{model_name}'")
            return await self.classifier_rerank(query, docs, return_docs, model_name)

        logger.info(f"EmbeddingService: Using Infinity engine for model '{model_name}' (no classifier available)")
        
        if not self.is_running:
            await self.start()
        
        logger.info(f"EmbeddingService: Calling engine_array['{model_name}'].rerank for {len(docs)} documents")
        scores, usage = await self.engine_array[model_name].rerank(
            query=query, docs=docs, raw_scores=False
        )
        docs_to_return: Optional[List[str]] = docs if return_docs else None
        return to_rerank_response(scores=scores, documents=docs_to_return, model=model_name, usage=usage)
