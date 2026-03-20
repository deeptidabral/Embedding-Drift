"""Data loader and preprocessor for the Sparkov Credit Card Fraud Detection dataset.

The Sparkov dataset contains ~1.8M simulated credit card transactions spanning
January 2019 through December 2020, with a realistic fraud rate of ~0.58%.
This module provides utilities for loading, cleaning, transforming, and sampling
the data for use in embedding-based drift detection experiments.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.embeddings.generator import LocalEmbeddingGenerator

logger = logging.getLogger(__name__)

# Expected columns in the raw Sparkov CSV files.
SPARKOV_COLUMNS = [
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "category",
    "amt",
    "first",
    "last",
    "gender",
    "street",
    "city",
    "state",
    "zip",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "unix_time",
    "merch_lat",
    "merch_long",
    "is_fraud",
]

# Amount band edges used for discretisation.
AMOUNT_BAND_EDGES = [0, 10, 50, 100, 250, 500, 1000, float("inf")]
AMOUNT_BAND_LABELS = [
    "micro",       # 0-10
    "small",       # 10-50
    "medium",      # 50-100
    "large",       # 100-250
    "high",        # 250-500
    "very_high",   # 500-1000
    "extreme",     # 1000+
]


class SparkovDataLoader:
    """Loader and preprocessor for the Sparkov fraud detection dataset.

    Typical usage::

        loader = SparkovDataLoader()
        df = loader.load("data/fraudTrain.csv")
        df = loader.preprocess(df)
        texts = loader.to_embedding_batch(df, text_column="transaction_text")
    """

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def load(path: Union[str, Path]) -> pd.DataFrame:
        """Load a Sparkov CSV file (fraudTrain.csv or fraudTest.csv).

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Raw dataframe with all original columns.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If required columns are missing from the file.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        logger.info("Loading Sparkov dataset from %s", path)
        df = pd.read_csv(path, low_memory=False)

        # Drop the unnamed index column that Sparkov CSVs include.
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        missing = set(SPARKOV_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV is missing expected Sparkov columns: {sorted(missing)}"
            )

        logger.info(
            "Loaded %d transactions (%d fraud, %.2f%% fraud rate)",
            len(df),
            df["is_fraud"].sum(),
            100 * df["is_fraud"].mean(),
        )
        return df

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich the raw dataframe with derived features.

        Derived columns added:
        - ``timestamp``: parsed datetime from ``trans_date_trans_time``
        - ``hour_of_day``: integer 0-23
        - ``day_of_week``: integer 0 (Monday) -- 6 (Sunday)
        - ``day_name``: human-readable day name (e.g. "Monday")
        - ``is_weekend``: boolean flag for Saturday/Sunday
        - ``amount_band``: categorical discretisation of ``amt``
        - ``customer_age``: approximate age at time of transaction
        - ``distance_to_merchant``: Haversine-approximated km between
          customer and merchant coordinates

        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe as returned by :meth:`load`.

        Returns
        -------
        pd.DataFrame
            Enriched copy of the dataframe.
        """
        df = df.copy()

        # Parse timestamps.
        df["timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Mon
        df["day_name"] = df["timestamp"].dt.day_name()
        df["is_weekend"] = df["day_of_week"].isin([5, 6])

        # Amount band.
        df["amount_band"] = pd.cut(
            df["amt"],
            bins=AMOUNT_BAND_EDGES,
            labels=AMOUNT_BAND_LABELS,
            right=False,
        )

        # Customer age (approximate, using dob and transaction date).
        df["dob_parsed"] = pd.to_datetime(df["dob"])
        df["customer_age"] = (
            (df["timestamp"] - df["dob_parsed"]).dt.days / 365.25
        ).astype(int)
        df = df.drop(columns=["dob_parsed"])

        # Distance between customer location and merchant location (km).
        df["distance_to_merchant"] = _haversine_km(
            df["lat"].values,
            df["long"].values,
            df["merch_lat"].values,
            df["merch_long"].values,
        )

        # Generate natural-language transaction text for embedding.
        df["transaction_text"] = df.apply(SparkovDataLoader.to_transaction_text, axis=1)

        logger.info(
            "Preprocessing complete. Shape: %s. Derived columns added.",
            df.shape,
        )
        return df

    # ------------------------------------------------------------------
    # Text conversion for embeddings
    # ------------------------------------------------------------------

    @staticmethod
    def to_transaction_text(row: pd.Series) -> str:
        """Convert a single transaction row into a natural-language description.

        The resulting text is designed to be fed to an embedding model so that
        semantically similar transactions cluster together.

        Parameters
        ----------
        row : pd.Series
            A single row from a preprocessed Sparkov dataframe.

        Returns
        -------
        str
            Human-readable transaction description.
        """
        # Safely extract fields, falling back to raw columns when derived
        # columns have not been added yet.
        timestamp = row.get("timestamp", row.get("trans_date_trans_time", ""))
        if isinstance(timestamp, pd.Timestamp):
            day_name = timestamp.day_name()
            time_str = timestamp.strftime("%H:%M")
        else:
            day_name = row.get("day_name", "")
            time_str = str(timestamp).split(" ")[-1][:5] if " " in str(timestamp) else ""

        cc_last4 = str(int(row["cc_num"]))[-4:]
        age = row.get("customer_age", "unknown")
        merchant = str(row["merchant"]).replace("fraud_", "")
        distance = row.get("distance_to_merchant")
        distance_str = f" Distance to merchant: {distance:.1f} km." if distance is not None else ""

        text = (
            f"Transaction of ${row['amt']:.2f} at {merchant} in category "
            f"{row['category']}, {row['city']}, {row['state']} on {day_name} "
            f"at {time_str}. Card ending {cc_last4}. Customer age {age}."
            f"{distance_str}"
        )
        return text

    @staticmethod
    def to_embedding_batch(
        df: pd.DataFrame,
        text_column: str = "transaction_text",
    ) -> list[str]:
        """Return a list of text strings suitable for batch embedding.

        If *text_column* already exists in *df*, its values are returned
        directly.  Otherwise :meth:`to_transaction_text` is applied row-wise
        and the resulting texts are returned.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe.
        text_column : str
            Name of the column containing pre-generated text.  If absent, texts
            are generated on the fly.

        Returns
        -------
        list[str]
            List of transaction description strings.
        """
        if text_column in df.columns:
            return df[text_column].tolist()

        logger.info("Generating transaction texts for %d rows", len(df))
        texts = df.apply(SparkovDataLoader.to_transaction_text, axis=1).tolist()
        return texts

    @staticmethod
    def generate_embeddings(
        df: pd.DataFrame,
        model: LocalEmbeddingGenerator | None = None,
        text_column: str = "transaction_text",
        batch_size: int = 256,
    ) -> np.ndarray:
        """Generate embedding vectors for every row in *df*.

        If *model* is ``None`` a :class:`LocalEmbeddingGenerator` is created
        internally (sentence-transformers, all-MiniLM-L6-v2, 384 dims).

        Parameters
        ----------
        df : pd.DataFrame
            Raw or preprocessed Sparkov dataframe.
        model : LocalEmbeddingGenerator or None
            Embedding model to use.  Defaults to a fresh local generator.
        text_column : str
            Column name that holds pre-generated transaction texts.  If the
            column does not exist, texts are generated via
            :meth:`to_transaction_text`.
        batch_size : int
            Number of texts encoded per forward pass.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(df), 384)`` (or whatever the model
            dimensionality is).
        """
        if model is None:
            model = LocalEmbeddingGenerator()

        # Build transaction texts if not already present.
        if text_column in df.columns:
            texts = df[text_column].tolist()
        else:
            logger.info("Generating transaction texts for %d rows", len(df))
            texts = df.apply(SparkovDataLoader.to_transaction_text, axis=1).tolist()

        n = len(texts)
        all_embeddings: list[np.ndarray] = []

        for start in range(0, n, batch_size):
            chunk = texts[start : start + batch_size]
            vecs = model.encode_texts(chunk)
            all_embeddings.append(np.asarray(vecs))

            if start > 0 and start % 1000 < batch_size:
                logger.info(
                    "generate_embeddings progress: %d / %d rows encoded",
                    start + len(chunk),
                    n,
                )

        result = np.concatenate(all_embeddings, axis=0)
        logger.info(
            "generate_embeddings complete: shape %s",
            result.shape,
        )
        return result

    @staticmethod
    def generate_transaction_texts(df: pd.DataFrame) -> pd.Series:
        """Convert every row in *df* to a transaction text string.

        This is a convenience wrapper around :meth:`to_transaction_text`
        that returns a :class:`pd.Series` (one text per row) so it can be
        passed directly to ``LocalEmbeddingGenerator.encode_texts``.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or preprocessed Sparkov dataframe.

        Returns
        -------
        pd.Series
            Series of transaction description strings, same index as *df*.
        """
        return df.apply(SparkovDataLoader.to_transaction_text, axis=1)

    # ------------------------------------------------------------------
    # Time-based splitting
    # ------------------------------------------------------------------

    @staticmethod
    def split_by_time(
        df: pd.DataFrame,
        reference_end_date: Union[str, datetime],
        production_start_date: Union[str, datetime],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into reference and production windows by timestamp.

        Transactions *before* ``reference_end_date`` form the reference
        distribution.  Transactions *on or after* ``production_start_date``
        form the production distribution.  Transactions between the two dates
        (the gap, if any) are excluded so that the windows are cleanly
        separated.

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe (must contain the ``timestamp`` column).
        reference_end_date : str or datetime
            End of the reference window (exclusive).
        production_start_date : str or datetime
            Start of the production window (inclusive).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(reference_df, production_df)``
        """
        if "timestamp" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'timestamp' column. "
                "Run preprocess() first."
            )

        ref_end = pd.Timestamp(reference_end_date)
        prod_start = pd.Timestamp(production_start_date)

        reference = df[df["timestamp"] < ref_end].copy()
        production = df[df["timestamp"] >= prod_start].copy()

        logger.info(
            "Split: reference=%d rows (< %s), production=%d rows (>= %s)",
            len(reference),
            ref_end.date(),
            len(production),
            prod_start.date(),
        )
        return reference, production

    # ------------------------------------------------------------------
    # Feature matrix for ML models
    # ------------------------------------------------------------------

    @staticmethod
    def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Extract a numerical feature matrix for ML modelling.

        The returned dataframe contains only numeric columns suitable for
        tree-based or linear models:

        - ``amt`` -- transaction amount in USD
        - ``hour_of_day`` -- 0-23
        - ``day_of_week`` -- 0 (Mon) to 6 (Sun)
        - ``is_weekend`` -- 0 or 1
        - ``city_pop`` -- city population
        - ``customer_age`` -- approximate age
        - ``lat``, ``long`` -- customer coordinates
        - ``merch_lat``, ``merch_long`` -- merchant coordinates
        - ``distance_to_merchant`` -- km (Haversine)

        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataframe.

        Returns
        -------
        pd.DataFrame
            Numeric feature matrix (no target column).
        """
        feature_cols = [
            "amt",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "city_pop",
            "customer_age",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "distance_to_merchant",
        ]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns (did you run preprocess()?): {missing}"
            )

        features = df[feature_cols].copy()
        features["is_weekend"] = features["is_weekend"].astype(int)
        return features

    # ------------------------------------------------------------------
    # Stratified sampling
    # ------------------------------------------------------------------

    @staticmethod
    def sample_stratified(
        df: pd.DataFrame,
        n: int,
        by: str = "category",
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """Draw a stratified sample preserving the distribution of *by*.

        If a stratum has fewer rows than its proportional allocation, all rows
        in that stratum are included.

        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe.
        n : int
            Desired total number of rows in the sample.
        by : str
            Column to stratify on (default ``"category"``).
        random_state : int or None
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Stratified subsample of *df*.
        """
        if by not in df.columns:
            raise ValueError(f"Stratification column '{by}' not in dataframe.")

        n = min(n, len(df))
        proportions = df[by].value_counts(normalize=True)
        samples: list[pd.DataFrame] = []

        for value, prop in proportions.items():
            stratum = df[df[by] == value]
            stratum_n = max(1, int(round(prop * n)))
            stratum_n = min(stratum_n, len(stratum))
            samples.append(
                stratum.sample(n=stratum_n, random_state=random_state)
            )

        result = pd.concat(samples, ignore_index=True)
        logger.info(
            "Stratified sample: requested %d, got %d (by='%s')",
            n,
            len(result),
            by,
        )
        return result


# --------------------------------------------------------------------------
# Private helpers
# --------------------------------------------------------------------------

def _haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised Haversine distance in kilometres."""
    r = 6371.0  # Earth radius in km
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    return 2 * r * np.arcsin(np.sqrt(a))
