"""Persist BenchmarkResult objects to/from a JSON file (merge + dedupe by name)."""

import json
import os
from dataclasses import asdict
from typing import List

from .benchmark import BenchmarkResult
from .config import RESULTS_FILE_PATH


def save_results_to_file(
    results: List[BenchmarkResult],
    file_path: str = RESULTS_FILE_PATH,
) -> None:
    """Merge results with existing file data (dedupe by name) and write back."""
    try:
        existing_results = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                existing_results = [BenchmarkResult(**r) for r in data]
            except Exception as read_err:
                print(f"Could not read existing file, starting fresh: {read_err}")

        merged_map = {r.name: r for r in existing_results}
        for r in results:
            merged_map[r.name] = r

        combined_results = list(merged_map.values())
        serialized_data = [asdict(r) for r in combined_results]

        with open(file_path, "w") as f:
            json.dump(serialized_data, f, indent=4)
        print(f"[Backup] Merged & saved {len(combined_results)} total unique results to '{file_path}'")
    except Exception as e:
        print(f"Failed to save results to file: {e}")


def load_results_from_file(
    file_path: str = RESULTS_FILE_PATH,
) -> List[BenchmarkResult]:
    """Load BenchmarkResult list back from a JSON file."""
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        return [BenchmarkResult(**r) for r in data]
    except Exception as e:
        print(f"Failed to load results from file: {e}")
        return []
