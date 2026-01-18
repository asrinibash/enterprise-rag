"""LLM response generation with OpenAI and Groq support."""

import logging
from typing import Optional, List
from langchain_core.documents import Document

from src.config import settings
from src.llm.prompts import PromptTemplates

logger = logging.getLogger(__name__)


class LLMGenerator:
    """Generate responses using LLM (OpenAI, Groq, or local)."""
    
    def __init__(
        self,
        provider: str = settings.LLM_PROVIDER,
        model_name: str = settings.LLM_MODEL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
    ):
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self.prompts = PromptTemplates()
        
        # Initialize based on provider
        if self.provider == "groq" and settings.GROQ_API_KEY:
            try:
                from groq import Groq
                self.client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info(f"Groq client initialized with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                logger.info("Install with: pip install groq")
        
        elif self.provider == "openai" and settings.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info(f"OpenAI client initialized with model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        else:
            logger.info("No LLM provider configured. Using fallback mode.")
    
    def generate(
        self,
        query: str,
        context_documents: List[Document],
        use_citations: bool = True,
    ) -> dict:
        """
        Generate answer using retrieved context.
        
        Returns:
            dict with 'answer', 'sources', and 'model_used'
        """
        # Format context from documents
        context = self.prompts.format_context(context_documents)
        
        # Generate prompt
        if use_citations:
            prompt = self.prompts.rag_prompt_with_citations(query, context)
        else:
            prompt = self.prompts.rag_prompt(query, context)
        
        # Generate response based on provider
        if self.client:
            try:
                if self.provider == "groq":
                    response = self._generate_groq(prompt)
                elif self.provider == "openai":
                    response = self._generate_openai(prompt)
                else:
                    response = self._generate_fallback(query, context_documents)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response = self._generate_fallback(query, context_documents)
        else:
            response = self._generate_fallback(query, context_documents)
        
        # Prepare sources
        sources = [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata,
            }
            for doc in context_documents
        ]
        
        return {
            "answer": response,
            "sources": sources,
            "model_used": f"{self.provider}:{self.model_name}" if self.client else "fallback",
        }
    
    def _generate_groq(self, prompt: str) -> str:
        """Generate response using Groq API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context. Be concise and accurate."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided context."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        return response.choices[0].message.content
    
    def _generate_fallback(
        self,
        query: str,
        documents: List[Document]
    ) -> str:
        """
        Fallback response when no LLM is available.
        Returns the most relevant document snippet.
        """
        logger.info("Using fallback response (no LLM available)")
        
        if not documents:
            return "No relevant information found in the knowledge base."
        
        # Return top document with a disclaimer
        top_doc = documents[0]
        source = top_doc.metadata.get("source", "Unknown source")
        
        response = f"""Based on the available information (from {source}):

{top_doc.page_content[:500]}...

[Note: This is a direct excerpt from the knowledge base. For a synthesized answer, please configure an LLM (set OPENAI_API_KEY in .env)]"""
        
        return response
    
    def is_llm_available(self) -> bool:
        """Check if LLM is properly configured."""
        return self.client is not None