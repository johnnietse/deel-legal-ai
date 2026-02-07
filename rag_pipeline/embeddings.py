# RAG Pipeline - Embeddings Module
"""
Vector embeddings generation using Google Gemini API.

Uses text-embedding-004 model for generating high-quality embeddings
suitable for legal document retrieval.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEMINI_API_KEY, GEMINI_API_BASE, GEMINI_EMBEDDING_MODEL

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    text: str
    embedding: List[float]
    model: str
    token_count: int


class GeminiEmbeddings:
    """
    Generate embeddings using Google Gemini API.
    
    Features:
    - Batch embedding for efficiency
    - Retry logic for API failures
    - Token counting
    - Rate limiting
    """
    
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_EMBEDDING_MODEL,
        batch_size: int = 100
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.base_url = GEMINI_API_BASE
        
        # Validate API key
        if not self.api_key:
            raise ValueError("Gemini API key is required")
    
    def _make_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a request to Gemini API"""
        url = f"{self.base_url}/{endpoint}"
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_text(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            EmbeddingResult with embedding vector
        """
        endpoint = f"models/{self.model}:embedContent"
        
        payload = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}]
            }
        }
        
        try:
            result = self._make_request(endpoint, payload)
            embedding = result.get("embedding", {}).get("values", [])
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.model,
                token_count=len(text.split())  # Approximate
            )
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def embed_batch(
        self, 
        texts: List[str], 
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of EmbeddingResult objects
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(texts, desc="Generating embeddings") if show_progress else texts
        
        for text in iterator:
            try:
                result = self.embed_text(text)
                results.append(result)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                # Create empty result to maintain alignment
                results.append(EmbeddingResult(
                    text=text,
                    embedding=[],
                    model=self.model,
                    token_count=0
                ))
        
        return results
    
    def embed_documents(
        self, 
        documents: List[Dict[str, Any]],
        text_key: str = "content"
    ) -> List[Dict[str, Any]]:
        """
        Embed documents and return with embeddings attached.
        
        Args:
            documents: List of documents with text content
            text_key: Key containing text to embed
            
        Returns:
            Documents with 'embedding' field added
        """
        texts = [doc.get(text_key, "") for doc in documents]
        embedding_results = self.embed_batch(texts)
        
        for doc, result in zip(documents, embedding_results):
            doc["embedding"] = result.embedding
            doc["embedding_model"] = result.model
        
        return documents


class GeminiChat:
    """
    Chat completion using Google Gemini API for RAG responses.
    """
    
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = "gemini-2.0-flash"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = GEMINI_API_BASE
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def generate(
        self, 
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate a response using Gemini.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        endpoint = f"models/{self.model}:generateContent"
        url = f"{self.base_url}/{endpoint}"
        
        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key
        }
        
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        if system_instruction:
            payload["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from response
        try:
            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        except (KeyError, IndexError) as e:
            logger.error(f"Error parsing Gemini response: {e}")
        
        return ""
    
    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate response with RAG context.
        
        Args:
            query: User query
            context: List of relevant context passages
            system_instruction: Optional system instruction
            
        Returns:
            Generated response with citations
        """
        default_instruction = """You are a legal research assistant for the Deel Lab for Global Employment. 
Your role is to provide accurate, well-cited answers based on the provided legal context.

When answering:
1. Base your response ONLY on the provided context
2. Cite specific sources when making claims
3. If the context doesn't contain enough information, say so
4. Be precise with legal terminology
5. Highlight any worker classification factors mentioned"""
        
        system = system_instruction or default_instruction
        
        # Format context
        context_text = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{ctx}" 
            for i, ctx in enumerate(context)
        ])
        
        prompt = f"""Based on the following legal context, answer the question.

LEGAL CONTEXT:
{context_text}

QUESTION: {query}

Provide a comprehensive answer with citations to the relevant sources."""
        
        return self.generate(prompt, system_instruction=system)


def test_embeddings():
    """Test the embeddings module"""
    embedder = GeminiEmbeddings()
    
    test_texts = [
        "The worker was classified as an independent contractor.",
        "Employment law in Ontario requires reasonable notice.",
        "The Sagaz test determines worker classification."
    ]
    
    print("\n" + "="*60)
    print("TESTING GEMINI EMBEDDINGS")
    print("="*60)
    
    for text in test_texts:
        try:
            result = embedder.embed_text(text)
            print(f"\n✅ Text: {text[:50]}...")
            print(f"   Embedding dimension: {len(result.embedding)}")
            print(f"   First 5 values: {result.embedding[:5]}")
        except Exception as e:
            print(f"\n❌ Error embedding text: {e}")


def test_chat():
    """Test the chat module"""
    chat = GeminiChat()
    
    print("\n" + "="*60)
    print("TESTING GEMINI CHAT")
    print("="*60)
    
    try:
        response = chat.generate_with_context(
            query="What factors determine if a worker is an employee?",
            context=[
                "In 671122 Ontario Ltd. v. Sagaz Industries Canada Inc., the Supreme Court established a four-factor test: control, ownership of tools, chance of profit, and risk of loss.",
                "The worker used company equipment exclusively and worked set hours determined by the employer."
            ]
        )
        print(f"\n✅ Response:\n{response}")
    except Exception as e:
        print(f"\n❌ Error generating response: {e}")


if __name__ == "__main__":
    test_embeddings()
    test_chat()
