from pathlib import Path
import pandas as pd

def load_excel(path: str | Path) -> pd.DataFrame:
    """Load the original Excel dataset."""
    return pd.read_excel(Path(path))

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load CSV (for prediction CLI)."""
    return pd.read_csv(Path(path))
