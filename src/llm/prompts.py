"""Prompt templates for LLM generation."""

from typing import List
from langchain_core.documents import Document


class PromptTemplates:
    """Collection of prompt templates for RAG."""
    
    @staticmethod
    def format_context(documents: List[Document]) -> str:
        """Format retrieved documents as context."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{doc.page_content}\n"
            )
        
        return "\n".join(context_parts)
    
    @staticmethod
    def rag_prompt(query: str, context: str) -> str:
        """Generate RAG prompt with context and query."""
        return f"""You are a helpful AI assistant. Answer the question based ONLY on the provided context. 

If the answer cannot be found in the context, say "I don't have enough information to answer this question based on the provided documents."

Be concise and accurate. Include relevant details from the context to support your answer.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
    
    @staticmethod
    def rag_prompt_with_citations(query: str, context: str) -> str:
        """Generate RAG prompt that encourages citations."""
        return f"""You are a helpful AI assistant. Answer the question based on the provided context.

Instructions:
1. Answer ONLY using information from the context provided
2. Cite your sources by mentioning the source number (e.g., "According to Source 1...")
3. If you cannot find the answer in the context, clearly state this
4. Be precise and include relevant details

CONTEXT:
{context}

QUESTION: {query}

ANSWER (with citations):"""
    
    @staticmethod
    def conversational_rag_prompt(
        query: str,
        context: str,
        chat_history: str = ""
    ) -> str:
        """Generate conversational RAG prompt with history."""
        history_section = ""
        if chat_history:
            history_section = f"""
CHAT HISTORY:
{chat_history}
"""
        
        return f"""You are a helpful AI assistant having a conversation with a user.

Answer the question based on the provided context and previous conversation history.

{history_section}

CONTEXT:
{context}

CURRENT QUESTION: {query}

ANSWER:"""
    
    @staticmethod
    def summarization_prompt(text: str, max_length: int = 200) -> str:
        """Generate summarization prompt."""
        return f"""Summarize the following text in approximately {max_length} words. 
Focus on the main points and key information.

TEXT:
{text}

SUMMARY:"""