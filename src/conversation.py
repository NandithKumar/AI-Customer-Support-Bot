"""
conversation.py — Multi-turn conversation history
Author: Paladugu Nandith Kumar
"""

from typing import List, Dict


class ConversationManager:
    def __init__(self, max_turns: int = 6):
        self.messages: List[Dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def format(self) -> str:
        recent = self.messages[-(self.max_turns * 2):]
        if not recent:
            return "No previous messages."
        return "\n".join(
            f"{'Customer' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in recent
        )

    def clear(self):
        self.messages = []
