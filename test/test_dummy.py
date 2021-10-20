"""
Write tests here
"""
import unittest


class TesTModel(unittest.TestCase):

    def test_dummy_pass(self):
        print("Dummy Test called successfully")
        self.assertTrue(True)

    def test_dummy_fail(self):
        self.assertTrue(False, "Dummy Tests is supposed to fail don't worry")
