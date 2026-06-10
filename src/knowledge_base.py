"""
knowledge_base.py — Loads documents and builds FAISS vector index
Author: Paladugu Nandith Kumar
"""

import os
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

SAMPLE_FAQ = """
# TechCorp Customer Support Knowledge Base

## Orders & Shipping
Q: How do I track my order?
A: Visit our website and click "Track Order" in the top menu. Enter your order number and email to see real-time updates. You'll also receive tracking emails after your order ships.

Q: How long does shipping take?
A: Standard: 5-7 business days. Express: 2-3 business days. Overnight: orders before 2PM EST. International: 10-15 business days.

Q: Can I change my shipping address?
A: Address changes are possible within 1 hour of placing your order. Contact support@techcorp.com immediately with your order number and new address.

Q: Do you offer free shipping?
A: Free standard shipping on orders over $50. Orders under $50 have a flat $5.99 shipping fee.

## Returns & Refunds
Q: What is your return policy?
A: We accept returns within 30 days of delivery. Items must be in original condition with all packaging. Start a return from your account dashboard.

Q: How long do refunds take?
A: Once we receive your return, refunds are processed in 3-5 business days. It appears on your payment method within 5-10 business days.

Q: Can I exchange an item?
A: Yes! Start an exchange the same way as a return. Select "Exchange" and choose your replacement item.

Q: What items cannot be returned?
A: Digital downloads, gift cards, Final Sale items, and customized/personalized items (unless defective).

## Account & Payments
Q: How do I reset my password?
A: Click "Forgot Password" on the login page, enter your email. You'll receive a reset link within 5 minutes. Check spam if it doesn't arrive.

Q: What payment methods do you accept?
A: Visa, Mastercard, American Express, Discover, PayPal, Apple Pay, Google Pay, and TechCorp Gift Cards.

Q: Is my payment information secure?
A: Yes, we use 256-bit SSL encryption and are PCI DSS compliant. We never store your full card number.

Q: How do I apply a discount code?
A: Enter the code in the "Promo Code" field at checkout. Only one code per order. Codes are case-insensitive.

## Products & Inventory
Q: How do I know if an item is in stock?
A: In-stock items show "Add to Cart". Out-of-stock items show "Notify Me" — click to get an email when back in stock.

Q: Do you price match?
A: Yes! Submit a price match request with a link to the competitor's price. Valid for 7 days after purchase.

## Technical Support
Q: How do I contact technical support?
A: Live chat (24/7), email techsupport@techcorp.com (2-hour response), or call 1-800-TECH-CORP (Mon-Fri 9AM-6PM EST).

Q: What is your warranty policy?
A: All products include a 1-year manufacturer warranty for defects. Extended warranties (2 or 3 years) available at checkout.

Q: How do I register my product for warranty?
A: Register at techcorp.com/warranty within 30 days of purchase using your order number and serial number.

## Business Hours & Contact
Q: What are your customer service hours?
A: Live chat: 24/7. Phone: Mon-Fri 9AM-6PM EST, Sat 10AM-4PM EST. Email: within 2 business hours on weekdays.

Q: Where are you located?
A: Headquarters in San Francisco, CA. Fulfillment centers across the US, Canada, and Europe.
"""


class KnowledgeBase:
    def __init__(self, chunk_size=400, chunk_overlap=40):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.documents: List[Document] = []
        self.store = None
        self.n_chunks = 0

    def load_sample(self):
        self.documents = self.splitter.create_documents(
            texts=[SAMPLE_FAQ],
            metadatas=[{"source": "TechCorp_FAQ"}]
        )

    def load_files(self, paths: List[str]):
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            text = self._load_pdf(path) if ext == ".pdf" else self._load_txt(path)
            chunks = self.splitter.create_documents(
                texts=[text],
                metadatas=[{"source": os.path.basename(path)}]
            )
            self.documents.extend(chunks)

    def _load_pdf(self, path):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)

    def _load_txt(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def build_index(self):
        self.store    = FAISS.from_documents(self.documents, self.embeddings)
        self.n_chunks = len(self.documents)

    def search(self, query: str, k: int = 4):
        return self.store.similarity_search_with_score(query, k=k)
