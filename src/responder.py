"""
responder.py — RAG response generator with smart escalation
Author: Paladugu Nandith Kumar
"""

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from src.knowledge_base import KnowledgeBase
from typing import Dict, Any

SUPPORT_PROMPT = ChatPromptTemplate.from_template("""
You are {bot_name}, an AI customer support assistant for {company}.
Tone: {tone}.

RULES:
- Answer ONLY from the knowledge base context
- If not in context, say you'll escalate to a human agent
- Keep responses under 4 sentences unless listing steps
- Always end with "Is there anything else I can help you with?"
- Show empathy first for complaints

KNOWLEDGE BASE:
{context}

HISTORY:
{history}

CUSTOMER: {question}

RESPONSE:
""")

ESCALATION_PROMPT = ChatPromptTemplate.from_template("""
You are {bot_name}, support assistant for {company}. Tone: {tone}.
The knowledge base doesn't have this info. Write a polite 2-sentence response:
1. Apologize for not having the info
2. Direct to support@{company_lower}.com or 1-800-SUPPORT

Customer question: {question}
""")

TONES = {
    "Professional & Helpful": "professional, helpful, solution-focused",
    "Friendly & Casual":      "friendly, warm, conversational",
    "Formal & Precise":       "formal, precise, structured",
}


class Responder:
    def __init__(self, kb: KnowledgeBase, model_name: str, api_key: str,
                 company: str = "TechCorp", bot_name: str = "Alex",
                 tone: str = "Professional & Helpful", top_k: int = 4):
        self.kb       = kb
        self.company  = company
        self.bot_name = bot_name
        self.top_k    = top_k
        self.tone_desc = TONES.get(tone, "helpful")

        self.llm = ChatGroq(
            model_name=model_name, groq_api_key=api_key,
            temperature=0.3, max_tokens=512
        )
        self.main_chain       = SUPPORT_PROMPT    | self.llm | StrOutputParser()
        self.escalation_chain = ESCALATION_PROMPT | self.llm | StrOutputParser()

    def respond(self, query: str, history: str = "") -> Dict[str, Any]:
        results = self.kb.search(query, k=self.top_k)
        relevant = [(doc, score) for doc, score in results if score < 1.8]
        context  = "\n\n---\n\n".join(d.page_content for d, _ in relevant)
        sources  = [d.page_content[:180] for d, _ in relevant]

        try:
            if context.strip():
                answer = self.main_chain.invoke({
                    "bot_name": self.bot_name, "company": self.company,
                    "tone": self.tone_desc, "context": context,
                    "history": history or "None", "question": query
                })
            else:
                answer = self.escalation_chain.invoke({
                    "bot_name": self.bot_name, "company": self.company,
                    "company_lower": self.company.lower(),
                    "tone": self.tone_desc, "question": query
                })
                sources = []
        except Exception as e:
            answer  = f"I'm experiencing a technical issue. Please contact support directly. ({e})"
            sources = []

        return {"answer": answer, "sources": sources}
