"""
test_support_bot.py — Unit tests for AI Customer Support Bot
Author: Paladugu Nandith Kumar
"""

import unittest
from src.intent_detector import IntentDetector
from src.conversation import ConversationManager


class TestIntentDetector(unittest.TestCase):
    def setUp(self):
        self.detector = IntentDetector()

    def test_order_tracking_intent(self):
        self.assertEqual(self.detector.detect("How do I track my order?"), "order_tracking")

    def test_returns_intent(self):
        self.assertEqual(self.detector.detect("I want to return my item"), "returns_refunds")

    def test_account_intent(self):
        self.assertEqual(self.detector.detect("I forgot my password"), "account")

    def test_payment_intent(self):
        self.assertEqual(self.detector.detect("What payment methods do you accept?"), "payment")

    def test_greeting_intent(self):
        self.assertEqual(self.detector.detect("Hello there!"), "greeting")

    def test_unknown_falls_back(self):
        result = self.detector.detect("xyzzy quantum flux capacitor")
        self.assertEqual(result, "general_inquiry")


class TestConversationManager(unittest.TestCase):
    def test_add_and_format(self):
        c = ConversationManager()
        c.add("user", "Hi")
        c.add("assistant", "Hello!")
        self.assertIn("Customer: Hi", c.format())
        self.assertIn("Assistant: Hello!", c.format())

    def test_clear(self):
        c = ConversationManager()
        c.add("user", "test")
        c.clear()
        self.assertEqual(c.format(), "No previous messages.")

    def test_max_turns_limits_history(self):
        c = ConversationManager(max_turns=2)
        for i in range(10):
            c.add("user", f"msg {i}")
            c.add("assistant", f"resp {i}")
        self.assertNotIn("msg 0", c.format())


if __name__ == "__main__":
    unittest.main()
