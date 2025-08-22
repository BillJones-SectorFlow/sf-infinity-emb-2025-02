"""
MXBAI reranker implemented as a HuggingFace classifier.
This class wraps a HuggingFace model and tokenizer that have been
fine-tuned for binary relevance classification.  It exposes synchronous
and asynchronous methods to score a list of documents against a query.
The asynchronous version offloads the synchronous ``rerank`` method to
the default event loop's executor.  This avoids blocking the event loop
while the model executes on the GPU.
"""
import asyncio
from typing import List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from create_mxbai_v2_reranker_prompt_template import (
    create_mxbai_v2_reranker_prompt_template,
)

class MXBAIRerankClassifier:
    """HuggingFace-based reranker for MXBAI v2 sequence models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        # Download tokenizer and model from HuggingFace.  The trust_remote_code
        # flag allows custom code to be executed during model loading.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Debug: Check initial tokenizer state
        print(f"Initial tokenizer pad_token: {self.tokenizer.pad_token}")
        print(f"Initial tokenizer eos_token: {self.tokenizer.eos_token}")
        print(f"Initial tokenizer pad_token_id: {getattr(self.tokenizer, 'pad_token_id', 'NOT_SET')}")
        
        # Set padding token if not already defined - this fixes the batch processing issue
        if self.tokenizer.pad_token is None:
            # Try multiple approaches to set padding token
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
            elif hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                print(f"Set pad_token to unk_token: {self.tokenizer.pad_token}")
            else:
                # Force add a padding token
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"Added custom pad_token: {self.tokenizer.pad_token}")
        
        # Debug: Check final tokenizer state
        print(f"Final tokenizer pad_token: {self.tokenizer.pad_token}")
        print(f"Final tokenizer pad_token_id: {getattr(self.tokenizer, 'pad_token_id', 'NOT_SET')}")
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=True
        ).eval()
        
        # Also ensure the model config has the padding token
        if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            print(f"Set model config pad_token_id: {self.model.config.pad_token_id}")
        # Determine target device: explicit override or best available
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def rerank(self, query: str, docs: List[str]) -> Tuple[List[float], int]:
        """Synchronously compute relevance scores for a list of documents.
        Args:
            query: The search query.
            docs: A list of documents to score.
        Returns:
            A tuple of (scores, usage) where scores is a list of floats
            representing the relevance probability for the "relevant" class
            (label "1"), and usage is the total number of tokens processed.
        """
        # Construct prompts for each document.  This duplicates the chat
        # templating used by Infinity's classifier endpoint.
        prompts = [create_mxbai_v2_reranker_prompt_template(query, d) for d in docs]
        
        try:
            # Try batch processing first
            print(f"Attempting batch processing for {len(prompts)} prompts...")
            enc = self.tokenizer(
                prompts, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            # Convert logits to probabilities and take the second column (relevant)
            scores = torch.softmax(logits, dim=-1)[:, 1].tolist()
            usage = int(enc.input_ids.numel())
            print(f"Batch processing successful!")
            return scores, usage
            
        except ValueError as e:
            if "padding token" in str(e):
                print(f"Batch processing failed due to padding token issue: {e}")
                print("Falling back to individual document processing...")
                
                # Fallback: process documents one by one
                all_scores = []
                total_usage = 0
                
                for i, prompt in enumerate(prompts):
                    print(f"Processing document {i+1}/{len(prompts)} individually...")
                    enc = self.tokenizer(
                        [prompt], padding=False, truncation=True, return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        logits = self.model(**enc).logits
                    score = torch.softmax(logits, dim=-1)[0, 1].item()
                    all_scores.append(score)
                    total_usage += int(enc.input_ids.numel())
                
                print(f"Individual processing completed successfully!")
                return all_scores, total_usage
            else:
                # Re-raise other ValueErrors
                raise
    
    async def arerank(self, query: str, docs: List[str]) -> Tuple[List[float], int]:
        """Asynchronously compute relevance scores using ``asyncio``.
        This method offloads the synchronous ``rerank`` computation to a
        background thread pool so that the event loop remains responsive.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.rerank(query, docs))
