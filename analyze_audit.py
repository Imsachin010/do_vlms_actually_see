"""
analyze_audit.py
Root entrypoint delegating to src/analyze_audit.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.analyze_audit import main

if __name__ == "__main__":
    main()
