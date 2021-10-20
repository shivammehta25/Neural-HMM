import os
import unittest

import torch
from pytorch_lightning.utilities import rank_zero_only


@rank_zero_only
def run_tests():
    with torch.no_grad():
        print("Running Tests....")
        loader = unittest.TestLoader()
        suite = loader.discover(os.path.join(os.getcwd(), 'test'))
        runner = unittest.TextTestRunner()
        runner.run(suite)
        print("Testing Completed! Check the summary above")


if __name__ == '__main__':
    run_tests()
