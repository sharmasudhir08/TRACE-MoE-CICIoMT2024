"""Puts src/ on the path for every test, so a test file can also be run on its own."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
