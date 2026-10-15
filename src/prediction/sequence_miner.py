"""
SequenceMiner — PrefixSpan Sequential Pattern Mining Module
Extracts frequent app usage subsequences from session logs and injects
them as binary feature flags into the ML pipeline.

Architecture:
  1. Segment event stream into sessions (gap > 30 min = new session)
  2. Run PrefixSpan on session sequences (min_support=3%)
  3. Extract top-K frequent patterns
  4. For each event: compute binary pattern-match features based on recent history
  5. Inject 10 binary flags + 1 confidence score = 11 new features

Reference: Pei et al., "PrefixSpan: Mining Sequential Patterns Efficiently by
           Prefix-Projected Pattern Growth", ICDE 2001.
"""
import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd


class SequenceMiner:
    """
    Mines frequent sequential app usage patterns from session data
    and creates binary match features for XGBoost.
    """

    def __init__(
        self,
        min_support: float = 0.03,
        max_pattern_length: int = 4,
        top_k_patterns: int = 10,
        session_gap_minutes: int = 30,
    ):
        """
        Args:
            min_support: Minimum fraction of sessions a pattern must appear in.
            max_pattern_length: Maximum length of mined patterns.
            top_k_patterns: Number of top patterns to use as features.
            session_gap_minutes: Gap in minutes to split into a new session.
        """
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.top_k_patterns = top_k_patterns
        self.session_gap_minutes = session_gap_minutes

        self.patterns: List[Tuple[List[str], int]] = []   # (pattern, support_count)
        self.pattern_support: List[float] = []             # fractional support per pattern
        self.total_sessions: int = 0
        self.is_fitted: bool = False
        self.feature_names: List[str] = []

    # ─────────────────────────────────────────────────────────────
    # Step 1: Session segmentation
    # ─────────────────────────────────────────────────────────────
    def segment_sessions(self, df: pd.DataFrame) -> List[List[str]]:
        """
        Split event DataFrame into sessions.
        A new session starts when gap > session_gap_minutes OR
        when user_id changes.

        Returns:
            List of sessions, each a list of app names (switch events only).
        """
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"] if "user_id" in df.columns
                            else ["timestamp"])

        sessions = []
        current_session = []
        prev_time = None
        prev_user = None
        gap_threshold = pd.Timedelta(minutes=self.session_gap_minutes)

        for _, row in df.iterrows():
            user = row.get("user_id", "single")
            ts = row["timestamp"]
            app = str(row["current_app"])

            # Start new session on user change or time gap
            if (prev_time is None or
                    ts - prev_time > gap_threshold or
                    user != prev_user):
                if len(current_session) >= 2:
                    sessions.append(current_session)
                current_session = []

            current_session.append(app)
            prev_time = ts
            prev_user = user

        if len(current_session) >= 2:
            sessions.append(current_session)

        return sessions

    # ─────────────────────────────────────────────────────────────
    # Step 2: PrefixSpan implementation (pure Python, no library dep)
    # ─────────────────────────────────────────────────────────────
    def _prefix_span(
        self, sequences: List[List[str]], min_count: int, prefix: List[str],
        results: List[Tuple[List[str], int]]
    ) -> None:
        """
        Recursive PrefixSpan algorithm.
        Mines frequent subsequences from the projected database.
        """
        if len(prefix) >= self.max_pattern_length:
            return

        # Count item frequencies in projected database
        item_counts: Dict[str, int] = defaultdict(int)
        for seq in sequences:
            seen_items = set()
            for item in seq:
                if item not in seen_items:
                    item_counts[item] += 1
                    seen_items.add(item)

        for item, count in item_counts.items():
            if count >= min_count:
                new_prefix = prefix + [item]
                results.append((new_prefix, count))

                # Build projected database: suffixes after first occurrence of item
                projected = []
                for seq in sequences:
                    try:
                        idx = seq.index(item)
                        suffix = seq[idx + 1:]
                        if suffix:
                            projected.append(suffix)
                    except ValueError:
                        continue

                if projected:
                    self._prefix_span(projected, min_count, new_prefix, results)

    def mine_patterns(self, df: pd.DataFrame) -> List[Tuple[List[str], float]]:
        """
        Run PrefixSpan on session data. Returns top-K patterns with support.

        Args:
            df: Event DataFrame.

        Returns:
            List of (pattern, support_fraction) sorted by support descending.
        """
        print("    Segmenting sessions...")
        sessions = self.segment_sessions(df)
        self.total_sessions = len(sessions)
        min_count = max(2, int(self.min_support * self.total_sessions))

        print(f"      {self.total_sessions:,} sessions, min_count={min_count} "
              f"({self.min_support*100:.1f}%)")

        print(f"    Running PrefixSpan (max_len={self.max_pattern_length})...")
        raw_results: List[Tuple[List[str], int]] = []
        self._prefix_span(sessions, min_count, [], raw_results)

        # Filter to only patterns of length >= 2 (single items not useful)
        raw_results = [(p, c) for p, c in raw_results if len(p) >= 2]

        # Sort by support descending
        raw_results.sort(key=lambda x: x[1], reverse=True)

        # Take top-K
        top_results = raw_results[:self.top_k_patterns]

        self.patterns = top_results
        self.pattern_support = [c / self.total_sessions for _, c in top_results]

        # Build feature names
        self.feature_names = []
        for i, (pattern, _) in enumerate(top_results):
            name = "seq_" + "_".join(p[:6] for p in pattern)  # truncate long names
            self.feature_names.append(name)
        self.feature_names.append("seq_best_confidence")

        self.is_fitted = True

        print(f"      Found {len(raw_results)} patterns total, "
              f"keeping top {len(top_results)}")
        print(f"\n    Top-{len(top_results)} Patterns:")
        for i, ((pattern, count), support) in enumerate(
                zip(top_results, self.pattern_support)):
            print(f"      {i+1:2}. {' -> '.join(pattern):<40} "
                  f"support={support*100:.1f}% ({count} sessions)")

        return list(zip([p for p, _ in top_results], self.pattern_support))

    # ─────────────────────────────────────────────────────────────
    # Step 3: Feature creation
    # ─────────────────────────────────────────────────────────────
    def create_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        For each event row, create binary pattern-match features based
        on its recent app history (prev_app_1, prev_app_2, prev_app_3).
        Vectorized implementation — no iterrows.

        Returns:
            Array of shape (n_events, top_k_patterns + 1).
            Columns: binary match per pattern + best_confidence float.
        """
        if not self.is_fitted:
            raise RuntimeError("Call mine_patterns() first.")

        n = len(df)
        n_feats = len(self.patterns) + 1
        X = np.zeros((n, n_feats), dtype=np.float32)

        # Extract history columns as string arrays (vectorized, no iterrows)
        col_map = {}
        for col in ["prev_app_3", "prev_app_2", "prev_app_1", "current_app"]:
            if col in df.columns:
                col_map[col] = df[col].fillna("").astype(str).values
            else:
                col_map[col] = np.full(n, "", dtype=object)

        # Stack into (n, 4) history matrix
        history_mat = np.column_stack([
            col_map["prev_app_3"],
            col_map["prev_app_2"],
            col_map["prev_app_1"],
            col_map["current_app"],
        ])  # shape: (n, 4)

        # For each pattern, check subsequence match across all rows
        for j, ((pattern, _), support) in enumerate(
                zip(self.patterns, self.pattern_support)):
            if not pattern:
                continue
            # Check if pattern is a subsequence of each row's history
            # Using set-based fast check for short patterns (len <= 4)
            match_flags = np.ones(n, dtype=bool)
            it_pos = np.zeros(n, dtype=int)
            pat_len = len(pattern)
            matched = np.zeros(n, dtype=int)

            for item in pattern:
                # For each row, find next occurrence of item at or after it_pos
                for col_i in range(4):
                    col_vals = history_mat[:, col_i]
                    # Rows where this column matches item AND we haven't matched yet
                    found = (col_vals == item) & (col_i >= it_pos) & (matched < pat_len)
                    it_pos = np.where(found & (matched < pat_len), col_i + 1, it_pos)
                    matched = np.where(found & (matched < pat_len), matched + 1, matched)

            matches = matched >= pat_len
            X[:, j] = matches.astype(np.float32)
            # Update best confidence where matched
            X[matches, -1] = np.maximum(X[matches, -1], support)

        return X

    def _is_subsequence(self, pattern: List[str], sequence: List[str]) -> bool:
        """Check if pattern is a subsequence of sequence."""
        if not pattern:
            return True
        it = iter(sequence)
        return all(item in it for item in pattern)

    def get_feature_names(self) -> List[str]:
        """Return feature names for the pattern match features."""
        return self.feature_names.copy()

    def get_pattern_summary(self) -> pd.DataFrame:
        """Return patterns as a tidy DataFrame."""
        if not self.is_fitted:
            return pd.DataFrame()
        rows = []
        for i, ((pattern, count), support) in enumerate(
                zip(self.patterns, self.pattern_support)):
            rows.append({
                "rank": i + 1,
                "pattern": " -> ".join(pattern),
                "length": len(pattern),
                "session_count": count,
                "support_pct": round(support * 100, 2),
            })
        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save mined patterns to JSON."""
        data = {
            "total_sessions": self.total_sessions,
            "min_support": self.min_support,
            "max_pattern_length": self.max_pattern_length,
            "top_k_patterns": self.top_k_patterns,
            "patterns": [
                {
                    "pattern": pattern,
                    "session_count": int(count),
                    "support_pct": round(supp * 100, 3),
                }
                for (pattern, count), supp in zip(self.patterns, self.pattern_support)
            ],
            "feature_names": self.feature_names,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"    Saved {len(self.patterns)} patterns to {path}")

    @classmethod
    def load(cls, path: str) -> "SequenceMiner":
        """Load patterns from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        miner = cls(
            min_support=data["min_support"],
            max_pattern_length=data["max_pattern_length"],
            top_k_patterns=data["top_k_patterns"],
        )
        miner.total_sessions = data["total_sessions"]
        miner.patterns = [
            (entry["pattern"], entry["session_count"])
            for entry in data["patterns"]
        ]
        miner.pattern_support = [
            entry["support_pct"] / 100 for entry in data["patterns"]
        ]
        miner.feature_names = data["feature_names"]
        miner.is_fitted = True
        return miner
