"""
intent_detector.py — Keyword-based intent classification
Author: Paladugu Nandith Kumar
"""

INTENTS = {
    "order_tracking":   ["track", "order", "where is", "shipping status", "delivered", "dispatch"],
    "returns_refunds":  ["return", "refund", "exchange", "money back", "send back"],
    "payment":          ["pay", "payment", "card", "invoice", "charge", "billing", "discount", "promo"],
    "technical":        ["not working", "error", "broken", "bug", "issue", "problem", "crash"],
    "product_info":     ["price", "available", "stock", "size", "color", "feature", "spec"],
    "account":          ["password", "login", "account", "sign in", "reset", "register"],
    "contact":          ["contact", "phone", "email", "hours", "speak to", "human", "agent"],
    "warranty":         ["warranty", "guarantee", "register product", "defect", "broken"],
    "complaint":        ["angry", "frustrated", "terrible", "worst", "complaint", "unacceptable"],
    "greeting":         ["hello", "hi", "hey", "good morning", "good afternoon", "howdy"],
    "thanks":           ["thank", "thanks", "appreciate", "helpful", "great"],
}


class IntentDetector:
    def detect(self, query: str) -> str:
        q = query.lower()
        scores = {}
        for intent, keywords in INTENTS.items():
            score = sum(1 for kw in keywords if kw in q)
            if score > 0:
                scores[intent] = score
        return max(scores, key=scores.get) if scores else "general_inquiry"
