"""Ensure the repository root is importable so tests can import the modules."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
