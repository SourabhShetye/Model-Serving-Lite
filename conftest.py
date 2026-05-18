"""
conftest.py (root-level)

Adds the project root to sys.path so pytest can resolve `app.*` imports
without requiring PYTHONPATH to be set explicitly in every environment.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))