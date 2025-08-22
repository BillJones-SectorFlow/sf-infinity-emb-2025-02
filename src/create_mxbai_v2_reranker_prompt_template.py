"""
Utilities to construct prompts for the MXBAI v2 reranker.

The Mixedbread AI reranker expects a chat-style prompt to evaluate the
relevance of a document given a query.  This helper constructs the prompt
accordingly.  See the model card for more details:
https://huggingface.co/michaelfeil/mxbai-rerank-large-v2-seq
"""

from typing import Optional


def create_mxbai_v2_reranker_prompt_template(
    query: str, document: str, instruction: str = ""
) -> str:
    """Format a query and document into a chat template understood by MXBAI v2.

    Args:
        query: The search query.
        document: The document to evaluate.
        instruction: Optional additional instruction to prepend.

    Returns:
        A formatted prompt string to send to the classifier model.
    """
    instruction = f"instruction: {instruction}\n" if instruction else ""
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    templated = (
        f"<|endoftext|><|im_start|>system\n{system_prompt}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}"
        f"query: {query}\n"
        f"document: {document}\n"
        "You are a search relevance expert who evaluates how well documents match search queries. "
        "Provide a binary relevance judgment (0 = not relevant, 1 = relevant).\n"
        "Relevance:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return templated
