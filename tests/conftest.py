"""Shared test configuration and fixtures."""

from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root for all tests
root = Path(__file__).parent.parent.resolve()
load_dotenv(root / ".env")
