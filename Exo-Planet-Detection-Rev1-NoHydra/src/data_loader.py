import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(file_path: str):
    """
    Load the Kepler cumulative.csv  (handles the comment rows NASA includes).
    Returns a DataFrame or None on failure.
    """
    logger.info(f"Loading data from: {file_path}")
    try:
        df = pd.read_csv(file_path, comment="#")
        if df.empty:
            logger.warning("DataFrame is empty with comment='#'.  Retrying without comment filter.")
            df = pd.read_csv(file_path)
        if df.empty:
            logger.error("Loaded an empty DataFrame.")
            return None
        logger.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        logger.error(
            f"File not found: {file_path}\n"
            "  → Place cumulative.csv inside the  data/  folder and retry."
        )
        return None
    except Exception as exc:
        logger.error(f"Unexpected error loading data: {exc}", exc_info=True)
        return None
