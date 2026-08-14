# RAG Pipeline - Embeddings Module
"""
Vector embeddings generation using Google Gemini API.
Chat completion using Google Gemini API or Groq (Llama3).

Uses gemini-embedding-001 model for generating high-quality embeddings
suitable for legal document retrieval.
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Always import requests for API calls
import requests

# Optional google-genai SDK for advanced features
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    GEMINI_API_KEY, GEMINI_API_BASE, GEMINI_EMBEDDING_MODEL, GEMINI_CHAT_MODEL,
    GEMINI_CHAT_MODEL_FALLBACKS,
    GROQ_API_KEY, GROQ_API_BASE, GROQ_CHAT_MODEL
)
from rag_pipeline.gemini_key_manager import (
    key_manager,
    GeminiRateLimitError,
    PoolExhaustedError,
)

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
    - Rate limiting with key rotation
    """
    
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_EMBEDDING_MODEL,
        batch_size: int = 100,
        key_manager=None,
    ):
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size
        self.base_url = GEMINI_API_BASE
        self.key_manager = key_manager
        
        # Validate API key (only if no key manager)
        if not self.key_manager and not self.api_key:
            raise ValueError("Gemini API key is required")
    
    def _get_key(self) -> str:
        """Get current API key, using key manager if available."""
        if self.key_manager:
            return self.key_manager.get_key()
        return self.api_key
    
    def _report_rate_limit(self):
        """Report rate limit to key manager if available."""
        if self.key_manager:
            self.key_manager.report_rate_limit()
    
    def _report_success(self):
        """Report success to key manager if available."""
        if self.key_manager:
            self.key_manager.report_success()
    
    @staticmethod
    def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 15.0) -> None:
        """Jittered exponential backoff sleep."""
        delay = min(cap, base * (2 ** attempt)) * (0.75 + 0.5 * random.random())
        time.sleep(delay)
    
    def _make_request(
        self, 
        endpoint: str, 
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a request to Gemini API with key rotation on rate limits."""
        last_error = None
        max_retries = 10 if self.key_manager else 5
        
        for attempt in range(max_retries):
            try:
                current_key = self._get_key()
            except PoolExhaustedError as e:
                raise GeminiRateLimitError(str(e)) from e

            url = f"{self.base_url}/{endpoint}"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": current_key,
            }
            
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    self._report_success()
                    return response.json()
                
                elif response.status_code == 429:
                    self._report_rate_limit()
                    logger.warning(f"Embedding key rate limited (attempt {attempt+1}), rotating key")
                    self._sleep_backoff(attempt)
                    last_error = "429 rate limited"
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Embedding request timeout (attempt {attempt+1}/{max_retries})")
                self._sleep_backoff(attempt, base=5.0)
                last_error = "Timeout"
            except Exception as e:
                if isinstance(e, GeminiRateLimitError):
                    raise
                logger.warning(f"Embedding request error (attempt {attempt+1}/{max_retries}): {e}")
                self._sleep_backoff(attempt, base=5.0)
                last_error = str(e)
        
        raise GeminiRateLimitError(
            f"Gemini embedding failed after {max_retries} retries: {last_error}"
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
        model: str = GEMINI_CHAT_MODEL,
        key_manager=None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = GEMINI_API_BASE
        self.key_manager = key_manager
    
    def _get_key(self) -> str:
        """Get current API key, using key manager if available."""
        km = self.key_manager or key_manager
        if km.key_count == 0:
            km = key_manager
        return km.get_key()
    
    def _report_rate_limit(self):
        """Report rate limit to key manager if available."""
        km = self.key_manager or key_manager
        if km.key_count == 0:
            km = key_manager
        km.report_rate_limit()
    
    def _report_success(self):
        """Report success to key manager if available."""
        km = self.key_manager or key_manager
        if km.key_count == 0:
            km = key_manager
        km.report_success()
    
    def _sleep_backoff(self, attempt: int, base: float = 1.0, cap: float = 15.0) -> None:
        """Jittered exponential backoff sleep."""
        delay = min(cap, base * (2 ** attempt)) * (0.75 + 0.5 * random.random())
        time.sleep(delay)
    
    def generate(
        self, 
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 10,
        ) -> str:
        """
        Generate a response using Gemini with key rotation on rate limits.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum retry attempts (default 10)
            
        Returns:
            Generated text response
        """
        km = self.key_manager or key_manager
        if km.key_count == 0:
            km = key_manager
        logger.debug(f"GeminiChat.generate using key manager with {km.key_count} keys (current: {km.current_index}/{km.key_count})")

        # Model rotation: configured model first, then fallbacks on 403.
        models = [self.model] + [
            m for m in GEMINI_CHAT_MODEL_FALLBACKS
            if m != self.model
        ]
        models = list(dict.fromkeys(models))

        last_error = None
        for model in models:
            endpoint = f"models/{model}:generateContent"

            if model != self.model:
                logger.warning(
                    f"GeminiChat falling back from {self.model} to {model}"
                )

            for attempt in range(max_retries):
                try:
                    current_key = self._get_key()
                except PoolExhaustedError as e:
                    raise GeminiRateLimitError(str(e)) from e

                key_masked = km.get_key_masked()
                url = f"{self.base_url}/{endpoint}"
                
                headers = {
                    "Content-Type": "application/json",
                    "x-goog-api-key": current_key,
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
                
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        # Extract text from response
                        candidates = result.get("candidates", [])
                        if candidates:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            if parts and parts[0].get("text"):
                                if model != self.model:
                                    self.model = model
                                    logger.warning(
                                        f"Pinned working Gemini model: {model}"
                                    )
                                self._report_success()
                                return parts[0].get("text", "")
                        # 200 with empty candidates → treat as retryable failure
                        last_error = "429 rate limited (keys rotated)"
                        logger.warning(f"Empty Gemini response (attempt {attempt+1}/{max_retries}), retrying...")
                        self._sleep_backoff(attempt)
                        continue
                    
                    elif response.status_code == 429:
                        self._report_rate_limit()
                        logger.warning(f"Key {key_masked} rate limited (attempt {attempt+1}), rotated key")
                        self._sleep_backoff(attempt)
                        last_error = "429 rate limited (keys rotated)"
                        continue

                    elif response.status_code == 403:
                        # Model not accessible with this key. Try next fallback.
                        last_error = (
                            f"403 Forbidden for model {model} "
                            f"(key {key_masked}); trying fallback model"
                        )
                        logger.warning(last_error)
                        break

                    else:
                        response.raise_for_status()
                        
                except requests.exceptions.Timeout:
                    logger.warning(f"Timeout (attempt {attempt+1}/{max_retries}), retrying...")
                    self._sleep_backoff(attempt, base=5.0)
                    last_error = "Timeout"
                except Exception as e:
                    if isinstance(e, GeminiRateLimitError):
                        raise
                    logger.warning(f"Error (attempt {attempt+1}/{max_retries}): {e}, retrying...")
                    self._sleep_backoff(attempt, base=5.0)
                    last_error = str(e)

        raise GeminiRateLimitError(
            f"Gemini generate failed after {len(models)} model(s) x {max_retries} "
            f"retries: {last_error}"
        )
    
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
        default_instruction = (
            "You are a legal research assistant for the Deel Lab for Global Employment. "
            "Your role is to provide accurate, well-cited answers based on the provided legal context.\n\n"
            "When answering:\n"
            "1. Base your response ONLY on the provided context\n"
            "2. Cite specific sources when making claims\n"
            "3. If the context doesn't contain enough information, say so\n"
            "4. Be precise with legal terminology\n"
            "5. Highlight any worker classification factors mentioned"
        )
        
        system = system_instruction or default_instruction
        
        # Format context
        context_text = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{ctx}" 
            for i, ctx in enumerate(context)
        ])
        
        prompt = (
            "Based on the following legal context, answer the question.\n\n"
            "LEGAL CONTEXT:\n"
            f"{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "Provide a comprehensive answer with citations to the relevant sources."
        )
        
        return self.generate(prompt, system_instruction=system)


class GroqChat:
    """
    Chat completion using Groq API (Llama3 models).
    
    Features:
    - OpenAI-compatible API
    - Fast inference (~300 tok/s on Llama3-70B)
    - Generous free tier
    - No key rotation needed (single key with high quota)
    """

    def __init__(
        self,
        api_key: str = GROQ_API_KEY,
        model: str = GROQ_CHAT_MODEL,
        base_url: str = GROQ_API_BASE,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        
        if not self.api_key:
            raise ValueError("Groq API key is required (set GROQ_API_KEY in .env)")

        # Mock key manager for compatibility with ResponseVerifier and other
        # components that expect a GeminiChat-like interface
        class _MockKeyManager:
            key_count = 1
            def get_key_masked(self): return "groq-key"
            def get_key(self): return api_key
            def report_rate_limit(self): pass
            def report_success(self): pass
        self.key_manager = _MockKeyManager()

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> str:
        """
        Generate a response using Groq.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            max_retries: Maximum retry attempts
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)

                if response.status_code == 200:
                    result = response.json()
                    choices = result.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    last_error = "Empty response"
                    continue

                elif response.status_code == 429:
                    logger.warning(f"Groq rate limited (attempt {attempt+1}/{max_retries})")
                    time.sleep(2 ** attempt)
                    last_error = "429 rate limited"
                    continue

                elif response.status_code == 401:
                    raise ValueError("Invalid Groq API key")

                else:
                    response.raise_for_status()

            except requests.exceptions.Timeout:
                logger.warning(f"Groq timeout (attempt {attempt+1}/{max_retries})")
                time.sleep(2 ** attempt)
                last_error = "Timeout"
            except Exception as e:
                logger.warning(f"Groq error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
                last_error = str(e)

        raise RuntimeError(f"Groq generate failed after {max_retries} retries: {last_error}")

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
        default_instruction = (
            "You are a legal research assistant for the Deel Lab for Global Employment. "
            "Your role is to provide accurate, well-cited answers based on the provided legal context.\n\n"
            "When answering:\n"
            "1. Base your response ONLY on the provided context\n"
            "2. Cite specific sources when making claims\n"
            "3. If the context doesn't contain enough information, say so\n"
            "4. Be precise with legal terminology\n"
            "5. Highlight any worker classification factors mentioned"
        )

        system = system_instruction or default_instruction

        # Format context
        context_text = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{ctx}"
            for i, ctx in enumerate(context)
        ])

        prompt = (
            "Based on the following legal context, answer the question.\n\n"
            "LEGAL CONTEXT:\n"
            f"{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "Provide a comprehensive answer with citations to the relevant sources."
        )

        return self.generate(prompt, system_instruction=system)


class MultiModelChat:
    """
    Chat completion with automatic multi-provider fallback.

    Tries providers in order (Groq first, then Gemini). If a provider
    fails (rate limit, timeout, auth, network), falls through to the
    next one. Only raises when ALL providers fail, so a single provider
    outage never degrades the user-facing answer.

    Exposes the same interface as GeminiChat/GroqChat (generate,
    generate_with_context, .model, .key_manager) so it can be dropped
    into LegalRAGQuery, ResponseVerifier, and MultiHopRetriever unchanged.
    """

    def __init__(self, providers: Optional[List[Any]] = None):
        self.providers = providers if providers is not None else self._default_providers()
        if not self.providers:
            raise ValueError(
                "No LLM providers configured. Set GROQ_API_KEY or GEMINI_API_KEY in .env"
            )
        # Compatibility surface: primary provider's model + key manager
        self.model = self.providers[0].model
        self.key_manager = getattr(self.providers[0], "key_manager", None)
        self.last_provider: Optional[str] = None
        self.last_error: Optional[str] = None

    @staticmethod
    def _default_providers() -> List[Any]:
        """Build provider chain: Groq (primary) → Gemini (fallback)."""
        providers: List[Any] = []
        try:
            providers.append(GroqChat())
            logger.info("MultiModelChat: Groq provider ready")
        except Exception as e:
            logger.warning(f"MultiModelChat: Groq unavailable ({e})")
        try:
            providers.append(GeminiChat())
            logger.info("MultiModelChat: Gemini fallback provider ready")
        except Exception as e:
            logger.warning(f"MultiModelChat: Gemini unavailable ({e})")
        return providers

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: Optional[int] = None,
    ) -> str:
        """
        Generate a response, falling back across providers on failure.

        Returns the first successful provider's answer. Raises RuntimeError
        only when every provider fails.
        """
        errors: List[str] = []
        for provider in self.providers:
            name = type(provider).__name__
            try:
                answer = provider.generate(
                    prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.last_provider = name
                self.last_error = None
                logger.info(f"MultiModelChat: answered via {name}")
                return answer
            except Exception as e:
                logger.warning(f"MultiModelChat: {name} failed ({e}), trying next provider")
                errors.append(f"{name}: {e}")

        self.last_error = "; ".join(errors)
        raise RuntimeError(
            f"All LLM providers failed: {self.last_error}"
        )

    def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_instruction: Optional[str] = None
    ) -> str:
        """
        Generate response with RAG context (provider-agnostic).
        """
        default_instruction = (
            "You are a legal research assistant for the Deel Lab for Global Employment. "
            "Your role is to provide accurate, well-cited answers based on the provided legal context.\n\n"
            "When answering:\n"
            "1. Base your response ONLY on the provided context\n"
            "2. Cite specific sources when making claims\n"
            "3. If the context doesn't contain enough information, say so\n"
            "4. Be precise with legal terminology\n"
            "5. Highlight any worker classification factors mentioned"
        )

        system = system_instruction or default_instruction

        context_text = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{ctx}"
            for i, ctx in enumerate(context)
        ])

        prompt = (
            "Based on the following legal context, answer the question.\n\n"
            "LEGAL CONTEXT:\n"
            f"{context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "Provide a comprehensive answer with citations to the relevant sources."
        )

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
