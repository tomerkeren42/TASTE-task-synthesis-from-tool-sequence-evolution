"""
Task set management: configuration, persistence, and loading.

Provides LengthDistributionConfig, TaskSetConfig, and TaskSetManager
for creating, saving, and reloading reusable task sets.
"""
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.common.domain_utils import ensure_tau2_path, WORKSPACE_ROOT
from src.common.sampler.length_distribution import LengthDistributionConfig

ensure_tau2_path()


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskSetConfig:
    """
    Full configuration for creating a task set.

    Fields:
        name: Human-readable label for this task set.
        domain: Domain name (e.g. "airline").
        seed_source: "original" | "empty" | file path to an existing tasks.json.
        num_tasks_to_generate: How many new tasks to create.
        length_distribution_config: How action sequence lengths are sampled.
        model_name: LLM model used for generation.
    """

    name: str
    domain: str
    seed_source: str  # "original" | "empty" | path
    num_tasks_to_generate: int
    length_distribution_config: LengthDistributionConfig
    model_name: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["length_distribution_config"] = self.length_distribution_config.to_dict()
        return d


# ---------------------------------------------------------------------------
# TaskSetManager
# ---------------------------------------------------------------------------

TASK_SETS_DIR = os.path.join(WORKSPACE_ROOT, "task_sets")


class TaskSetManager:
    """
    Manages saving reusable task sets.

    Each task set is stored as a directory under ``task_sets/`` containing:
        - tasks.json   – list of task dicts (seed + generated)
        - config.json  – the TaskSetConfig used to create the set
        - metadata.json – creation timestamp, counts, length stats
    """

    def __init__(self, base_dir: str = TASK_SETS_DIR):
        self.base_dir = base_dir

    # -- public API ---------------------------------------------------------

    def save(
        self,
        name: str,
        all_tasks: List[Dict[str, Any]],
        config: TaskSetConfig,
        num_seed_tasks: int = 0,
        num_generated_tasks: int = 0,
    ) -> str:
        """
        Persist a task set to disk.

        Args:
            name: Directory name under task_sets/.
            all_tasks: List of task dicts (seed + generated).
            config: The configuration that produced this set.
            num_seed_tasks: Number of seed tasks included.
            num_generated_tasks: Number of newly generated tasks.

        Returns:
            Absolute path to the saved task set directory.
        """
        task_set_dir = os.path.join(self.base_dir, name)
        os.makedirs(task_set_dir, exist_ok=True)

        # tasks.json  (strip keys that shouldn't be persisted)
        cleaned_tasks = [self._clean_task_dict(t) for t in all_tasks]
        tasks_path = os.path.join(task_set_dir, "tasks.json")
        with open(tasks_path, "w") as f:
            json.dump(cleaned_tasks, f, indent=4)

        # config.json
        config_path = os.path.join(task_set_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=4)

        # metadata.json
        lengths = self._extract_lengths(all_tasks)
        metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "num_tasks": len(all_tasks),
            "num_seed_tasks": num_seed_tasks,
            "num_generated_tasks": num_generated_tasks,
            "length_stats": self._length_stats(lengths) if lengths else {},
        }
        metadata_path = os.path.join(task_set_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        return task_set_dir

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _clean_task_dict(task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove keys that should not appear in the persisted tasks.json.

        Strips:
            - ``ticket`` (top-level)

        Preserved:
            - ``env_assertions`` (load-bearing for telecom; null for airline/retail)
            - ``reward_basis`` (telecom uses [ENV_ASSERTION]; airline/retail use default)
        """
        task = dict(task)  # shallow copy
        task.pop("ticket", None)
        return task

    def _resolve_tasks_path(self, name_or_path: str) -> str:
        """Resolve a name or path to an actual tasks.json file path."""
        # Direct file path
        if os.path.isfile(name_or_path):
            return name_or_path

        # Task-set name -> task_sets/<name>/tasks.json
        candidate = os.path.join(self.base_dir, name_or_path, "tasks.json")
        if os.path.isfile(candidate):
            return candidate

        raise FileNotFoundError(
            f"Cannot find tasks at '{name_or_path}'. "
            f"Checked as file path and as task set name under {self.base_dir}/."
        )

    @staticmethod
    def _extract_lengths(tasks: List[Dict[str, Any]]) -> List[int]:
        """Extract action sequence lengths from task dicts."""
        lengths = []
        for t in tasks:
            actions = t.get("evaluation_criteria", {}).get("actions", [])
            lengths.append(len(actions))
        return lengths

    @staticmethod
    def _length_stats(lengths: List[int]) -> Dict[str, Any]:
        """Compute summary statistics for a list of lengths."""
        if not lengths:
            return {}
        import numpy as np

        arr = np.array(lengths)
        return {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": round(float(arr.mean()), 2),
            "median": round(float(np.median(arr)), 2),
            "std": round(float(arr.std()), 2),
            "p10": round(float(np.percentile(arr, 10)), 2),
            "p90": round(float(np.percentile(arr, 90)), 2),
            "distribution": {str(l): int(c) for l, c in zip(*np.unique(arr, return_counts=True))},
        }
