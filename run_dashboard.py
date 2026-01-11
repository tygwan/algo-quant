#!/usr/bin/env python3
"""Launch Algo-Quant Dashboard."""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).parent / "src" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"Error: {app_path} not found")
        sys.exit(1)
    
    print("🚀 Starting Algo-Quant Dashboard...")
    print("   Open http://localhost:8501 in your browser")
    print("   Press Ctrl+C to stop\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port=8501",
            "--browser.gatherUsageStats=false",
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")


if __name__ == "__main__":
    main()
