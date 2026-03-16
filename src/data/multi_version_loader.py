"""
HAI Multi-Version Dataset Loader

Loads and harmonises all 5 HAI dataset releases into a unified DataFrame:
  - hai-20.07  (64 cols, semicolon, embedded attack col)
  - hai-21.03  (84 cols, comma, embedded attack col)
  - hai-22.04  (88 cols, comma, embedded Attack col)
  - hai-23.05  (87 cols, comma, separate label files)
  - haiend-23.05 (226 cols, comma, separate label files, DCS-expanded)

Each version is loaded with its 38 common sensor features (intersection of
hai-20.07 through hai-23.05).  haiend is loaded separately with its own 225
DCS features.  All returned DataFrames have a binary `Attack` column (0/1).
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import logger

# ---------------------------------------------------------------------------
# Version registry
# ---------------------------------------------------------------------------

VERSION_INFO = {
    "hai-20.07": {
        "root":       "hai-20.07/hai-20.07",
        "sep":        ";",
        "time_col":   "time",
        "train_glob": "train*.csv",
        "test_glob":  "test*.csv",
        "attack_mode": "embedded",        # attack column(s) inside data file
        "attack_col":  "attack",          # primary binary col
        "attack_subcols": ["attack_P1", "attack_P2", "attack_P3"],
    },
    "hai-21.03": {
        "root":       "hai-21.03/hai-21.03",
        "sep":        ",",
        "time_col":   "time",
        "train_glob": "train*.csv",
        "test_glob":  "test*.csv",
        "attack_mode": "embedded",
        "attack_col":  "attack",
        "attack_subcols": ["attack_P1", "attack_P2", "attack_P3"],
    },
    "hai-22.04": {
        "root":       "hai-22.04/hai-22.04",
        "sep":        ",",
        "time_col":   "timestamp",
        "train_glob": "train*.csv",
        "test_glob":  "test*.csv",
        "attack_mode": "embedded",
        "attack_col":  "Attack",
        "attack_subcols": [],
    },
    "hai-23.05": {
        "root":        "hai-23.05/hai-23.05",
        "sep":         ",",
        "time_col":    "timestamp",
        "train_glob":  "hai-train*.csv",
        "test_glob":   "hai-test*.csv",
        "attack_mode": "label_file",      # separate label CSV
        "label_glob":  "label-test*.csv",
        "label_time":  "timestamp",
        "label_col":   "label",
    },
    "haiend-23.05": {
        "root":        "haiend-23.05/haiend-23.05",
        "sep":         ",",
        "time_col":    "Timestamp",
        "train_glob":  "end-train*.csv",
        "test_glob":   "end-test*.csv",
        "attack_mode": "label_file",
        "label_glob":  "label-test*.csv",
        "label_time":  "timestamp",
        "label_col":   "label",
    },
}

# Sensor features common across hai-20.07 → hai-23.05 (38 features)
# Derived by taking the intersection of sensor columns in all four versions.
COMMON_FEATURES = [
    "P1_FCV01D", "P1_FCV01Z", "P1_FCV02D", "P1_FCV02Z",
    "P1_FCV03D", "P1_FCV03Z", "P1_FT01",   "P1_FT01Z",
    "P1_FT02",   "P1_FT02Z",  "P1_FT03",   "P1_FT03Z",
    "P1_LCV01D", "P1_LCV01Z", "P1_LIT01",
    "P2_24V",    "P2_ATSV",   "P2_CO",     "P2_STS",    "P2_TT01",
    "P3_FT01",   "P3_LCV01D", "P3_LCV01Z", "P3_LIT01",
    "P3_MV01",   "P3_MV02",   "P3_MV03",   "P3_MV04",
    "P3_PCV01D", "P3_PCV01Z", "P3_PIT01",  "P3_PIT02",
    "P4_HT01",   "P4_HT02",   "P4_LIT01",  "P4_MFM01",
    "P4_ST_FT01","P4_TT01",
]


def _glob_sorted(directory: Path, pattern: str) -> List[Path]:
    """Return sorted list of files matching *pattern* in *directory*."""
    files = sorted(directory.glob(pattern))
    return files


def _load_embedded_labels(df: pd.DataFrame, info: dict) -> pd.Series:
    """Extract binary attack label from embedded column."""
    col = info["attack_col"]
    if col not in df.columns:
        # fallback to any attack_* column
        candidates = [c for c in df.columns if "attack" in c.lower()]
        col = candidates[0] if candidates else None
    if col is None:
        return pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
    return (df[col].fillna(0).astype(float) > 0).astype(np.int8)


def _load_label_files(data_dir: Path, info: dict) -> Optional[pd.DataFrame]:
    """
    Load separate label CSVs and return a DataFrame indexed by timestamp.
    Returns None if no label files are found (train split has no attacks).
    """
    label_glob = info.get("label_glob", "label-test*.csv")
    label_files = _glob_sorted(data_dir, label_glob)
    if not label_files:
        return None
    parts = []
    for lf in label_files:
        ldf = pd.read_csv(lf)
        parts.append(ldf)
    labels = pd.concat(parts, ignore_index=True)
    labels = labels.rename(columns={
        info["label_time"]: "timestamp",
        info["label_col"]:  "label",
    })
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], format="mixed")
    labels = labels.set_index("timestamp")
    # Drop duplicate timestamps (keep last), so .get() always returns a scalar
    labels = labels[~labels.index.duplicated(keep="last")]
    return labels


def _harmonise_columns(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Keep only *features* that exist in df; fill missing with 0."""
    out = pd.DataFrame(index=df.index)
    for col in features:
        if col in df.columns:
            out[col] = df[col].values
        else:
            out[col] = 0.0
    return out


class MultiVersionLoader:
    """
    Loads and merges all HAI dataset versions into a single, harmonised
    training corpus for the diffusion model.

    Usage
    -----
    loader = MultiVersionLoader(hai_root="C:/path/to/HAI")
    X_train, y_train = loader.load_all(split="train", max_rows_per_version=200_000)
    X_end,   y_end   = loader.load_version("haiend-23.05", split="train")
    """

    def __init__(self, hai_root: str):
        self.hai_root = Path(hai_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_version(
        self,
        version: str,
        split: str = "train",
        features: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load a single HAI version.

        Parameters
        ----------
        version : one of VERSION_INFO keys
        split   : "train" or "test"
        features: list of columns to keep (None → COMMON_FEATURES)
        max_rows: max rows to load (None → all)

        Returns
        -------
        X : DataFrame of sensor readings
        y : ndarray of binary attack labels (int8)
        """
        info = VERSION_INFO[version]
        data_dir = self.hai_root / info["root"]

        glob_pattern = info["train_glob"] if split == "train" else info["test_glob"]
        files = _glob_sorted(data_dir, glob_pattern)

        if not files:
            logger.warning(f"[{version}] No {split} files found in {data_dir}")
            return pd.DataFrame(), np.array([], dtype=np.int8)

        logger.info(f"[{version}] Loading {len(files)} {split} file(s) from {data_dir}")

        parts_X, parts_y = [], []
        rows_loaded = 0

        # For label-file versions, load labels indexed by timestamp
        ext_labels = None
        if info["attack_mode"] == "label_file":
            ext_labels = _load_label_files(data_dir, info)

        for fpath in files:
            if max_rows is not None and rows_loaded >= max_rows:
                break

            nrows = None if max_rows is None else max_rows - rows_loaded
            df = pd.read_csv(fpath, sep=info["sep"], low_memory=False, nrows=nrows)
            logger.info(f"  {fpath.name}: {len(df):,} rows")

            # Parse timestamp
            tc = info["time_col"]
            if tc in df.columns:
                df[tc] = pd.to_datetime(df[tc], errors="coerce")
                df = df.set_index(tc)
            else:
                df.index = pd.RangeIndex(rows_loaded, rows_loaded + len(df))

            # Extract attack label
            if info["attack_mode"] == "embedded":
                y_part = _load_embedded_labels(df, info)
            else:
                if ext_labels is not None and isinstance(df.index, pd.DatetimeIndex):
                    # Align label series to data timestamps
                    label_series = ext_labels["label"]
                    y_vals = np.array(
                        [int(label_series.at[ts]) if ts in label_series.index else 0
                         for ts in df.index],
                        dtype=np.int8,
                    )
                    y_part = pd.Series(y_vals, index=df.index)
                else:
                    # Training files → all normal (no attack label files for train)
                    y_part = pd.Series(
                        np.zeros(len(df), dtype=np.int8), index=df.index
                    )

            # Select sensor features
            feat_list = features if features else COMMON_FEATURES
            X_part = _harmonise_columns(df, feat_list)

            parts_X.append(X_part)
            parts_y.append(y_part)
            rows_loaded += len(df)

        if not parts_X:
            return pd.DataFrame(), np.array([], dtype=np.int8)

        X = pd.concat(parts_X)
        y = pd.concat(parts_y).values.astype(np.int8)

        # Basic cleaning
        X = X.ffill().fillna(0).astype(np.float32)

        attack_rate = y.mean() * 100
        logger.info(
            f"[{version}] Loaded {len(X):,} rows | "
            f"{int(y.sum())} attacks ({attack_rate:.2f}%)"
        )
        return X, y

    def load_all(
        self,
        versions: Optional[List[str]] = None,
        split: str = "train",
        features: Optional[List[str]] = None,
        max_rows_per_version: Optional[int] = None,
        include_haiend: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and concatenate multiple HAI versions.

        Parameters
        ----------
        versions            : list of version names; defaults to standard 4
        split               : "train" or "test"
        features            : columns to keep (None → COMMON_FEATURES)
        max_rows_per_version: row cap per version
        include_haiend      : if True, load haiend-23.05 with its own features

        Returns
        -------
        X : float32 ndarray (n_total, n_features)
        y : int8   ndarray (n_total,)
        """
        if versions is None:
            versions = ["hai-20.07", "hai-21.03", "hai-22.04", "hai-23.05"]

        all_X, all_y = [], []

        for ver in versions:
            X_v, y_v = self.load_version(
                ver, split=split, features=features, max_rows=max_rows_per_version
            )
            if len(X_v):
                all_X.append(X_v.values)
                all_y.append(y_v)

        if include_haiend:
            # haiend has its own unique feature set
            X_end, y_end = self.load_haiend(split=split, max_rows=max_rows_per_version)
            # Only include if shapes match (pad/truncate to COMMON_FEATURES size)
            # For unified training we drop haiend sensor columns not in COMMON_FEATURES
            # and pad with zeros if needed.
            n_feat = len(features) if features else len(COMMON_FEATURES)
            if X_end.shape[1] >= n_feat:
                all_X.append(X_end[:, :n_feat])
            else:
                pad = np.zeros((len(X_end), n_feat - X_end.shape[1]), dtype=np.float32)
                all_X.append(np.hstack([X_end, pad]))
            all_y.append(y_end)

        if not all_X:
            return np.zeros((0, len(features or COMMON_FEATURES)), dtype=np.float32), \
                   np.zeros(0, dtype=np.int8)

        X_out = np.vstack(all_X).astype(np.float32)
        y_out = np.concatenate(all_y).astype(np.int8)

        logger.info(
            f"Combined dataset: {X_out.shape[0]:,} rows x {X_out.shape[1]} features | "
            f"{int(y_out.sum())} attacks ({y_out.mean()*100:.2f}%)"
        )
        return X_out, y_out

    def load_haiend(
        self,
        split: str = "train",
        max_rows: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load haiend-23.05 with its full 225-feature DCS-expanded schema.

        Returns
        -------
        X : float32 ndarray (n, 225)
        y : int8   ndarray (n,)
        """
        info = VERSION_INFO["haiend-23.05"]
        data_dir = self.hai_root / info["root"]

        # Discover all sensor columns
        glob_pat = info["train_glob"] if split == "train" else info["test_glob"]
        files = _glob_sorted(data_dir, glob_pat)
        if not files:
            return np.zeros((0, 225), dtype=np.float32), np.zeros(0, dtype=np.int8)

        first = pd.read_csv(files[0], sep=info["sep"], nrows=0)
        tc = info["time_col"]
        sensor_cols = [c for c in first.columns if c != tc]

        X_df, y = self.load_version(
            "haiend-23.05", split=split, features=sensor_cols, max_rows=max_rows
        )
        return X_df.values.astype(np.float32), y


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def load_graph_topology(graph_dir: str) -> Dict:
    """
    Load all DCS graph JSON files (Python-dict format) and the physical
    boiler graph (standard JSON), returning a dict with:
      {
        "dcs_nodes":  {module_id: [node_id, ...], ...},
        "dcs_edges":  {module_id: [(src, tgt, label), ...], ...},
        "phy_nodes":  [node_id, ...],
        "phy_edges":  [(src, tgt, ...), ...],
        "signal_names": [str, ...],    # edge labels = sensor/signal names
      }
    """
    graph_path = Path(graph_dir)
    dcs_nodes: Dict[str, List] = {}
    dcs_edges: Dict[str, List] = {}
    signal_names: List[str] = []

    for fpath in sorted(graph_path.glob("dcs_*.json")):
        module = fpath.stem  # e.g. "dcs_1001h"
        with open(fpath) as f:
            raw = f.read()
        # Files are Python-dict format (single quotes, True/False)
        try:
            g = ast.literal_eval(raw)
        except Exception:
            # Fallback: replace Python literals and parse as JSON
            raw_json = (
                raw.replace("True", "true").replace("False", "false").replace("None", "null")
                   .replace("'", '"')
            )
            g = json.loads(raw_json)

        nodes = [n["id"] for n in g.get("nodes", [])]
        edges = [(e["source"], e["target"], e.get("label", ""))
                 for e in g.get("links", [])]
        dcs_nodes[module] = nodes
        dcs_edges[module] = edges
        signal_names.extend(e[2] for e in edges if e[2])

    # Physical boiler graph (standard JSON)
    phy_boiler = graph_path / "phy_boiler.json"
    phy_nodes, phy_edges = [], []
    if phy_boiler.exists():
        with open(phy_boiler) as f:
            g = json.load(f)
        phy_nodes = [n["id"] for n in g.get("nodes", [])]
        phy_edges = [(e["source"], e["target"])
                     for e in g.get("links", [])]

    signal_names = sorted(set(signal_names))
    logger.info(
        f"Graph topology loaded: {len(dcs_nodes)} DCS modules, "
        f"{len(phy_nodes)} physical nodes, "
        f"{len(signal_names)} unique signals"
    )
    return {
        "dcs_nodes":    dcs_nodes,
        "dcs_edges":    dcs_edges,
        "phy_nodes":    phy_nodes,
        "phy_edges":    phy_edges,
        "signal_names": signal_names,
    }


def graph_feature_groups(topology: Dict, feature_names: List[str]) -> Dict[str, List[int]]:
    """
    Map graph signal names to feature indices.

    Returns dict:  {signal_name: [feature_index, ...]}
    Only signals whose name is a substring of a feature column are matched.
    """
    groups: Dict[str, List[int]] = {}
    for sig in topology["signal_names"]:
        idxs = [i for i, fn in enumerate(feature_names)
                if sig.lower() in fn.lower() or fn.lower() in sig.lower()]
        if idxs:
            groups[sig] = idxs
    logger.info(f"Mapped {len(groups)} graph signals to feature indices")
    return groups
