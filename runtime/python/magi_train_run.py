"""Runs a Magic tab training plan. Started by the app; see magi_train/__init__.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from magi_train.train import main  # noqa: E402  (the path must be set up first)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
