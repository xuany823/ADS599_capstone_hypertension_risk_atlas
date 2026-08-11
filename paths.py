from pathlib import Path
import sys

def get_project_root() -> Path:
    """Get the absolute path to the project root reliably."""
    return Path(__file__).resolve().parent

# Define standard project paths
BASE_PATH = get_project_root()
DATA_DIR = BASE_PATH / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
VALIDATION_DIR = BASE_PATH / "data" / "validation"
DATA_FINAL = DATA_DIR / "final"

def validate_and_alert(file_path, script_name, instructions):
    """
    Checks if a file exists. If not, prints specific instructions and exits.
    """
    if not Path(file_path).exists():
        print("!" * 60)
        print(f"CRITICAL: Missing file for {script_name}!")
        print(f"Expected path: {file_path}")
        print("-" * 60)
        print("INSTRUCTIONS TO RETRIEVE DATA:")
        print(instructions)
        print("!" * 60)
        sys.exit(1)
    else:
        print(f"✅ Data found for {script_name}. Proceeding...")