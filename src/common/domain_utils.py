"""
Utility functions for loading domain resources (policy, tasks, DB samples).

Note: For tool_spec, use ToolsSpecRetriever from tool_spec_retriever module.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Repository root (parent of ``src/``).  Single source of truth: every other
# module should import ``WORKSPACE_ROOT`` from here rather than recompute it
# from ``__file__`` (which breaks whenever a file is moved between depths).
# This file lives at ``<repo>/src/common/domain_utils.py`` so we walk up
# three levels.
WORKSPACE_ROOT = str(Path(__file__).resolve().parents[2])
_TAU2_BENCH_ROOT = os.path.join(WORKSPACE_ROOT, "tau2-bench")
_TAU2_BENCH_SRC = os.path.join(_TAU2_BENCH_ROOT, "src")
_ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, "artifacts")
_DOMAINS_DIR = os.path.join(_ARTIFACTS_DIR, "domains")


def ensure_tau2_path() -> None:
    """Ensure tau2-bench/src is on sys.path so ``import tau2`` works.

    Safe to call multiple times – the path is only inserted once.
    Other modules should call this (or import domain_utils, which calls it
    at module level) instead of duplicating the sys.path manipulation.
    """
    if _TAU2_BENCH_SRC not in sys.path:
        sys.path.insert(0, _TAU2_BENCH_SRC)


# Run at import time so that ``from domain_utils import …`` is sufficient
# to make tau2 importable.
ensure_tau2_path()


# Central dictionary of domain -> paths. Used instead of passing paths from callers.
# Add new domains here when extending support.
DOMAIN_PATHS: Dict[str, Dict[str, str]] = {}
for _domain in ("airline", "retail", "telecom"):
    _domain_dir = os.path.join(_DOMAINS_DIR, _domain)
    DOMAIN_PATHS[_domain] = {
        "base": _domain_dir,
        "policy": os.path.join(_domain_dir, "policy.md"),
        "tasks": os.path.join(_domain_dir, "tasks.json"),
    }

from tau2.data_model.tasks import Task  # type: ignore


def load_policy(domain: str) -> str:
    """
    Load policy from file.

    Args:
        domain: Domain name (must exist in DOMAIN_PATHS)

    Returns:
        Policy content as string
    """
    policy_path = DOMAIN_PATHS[domain]["policy"]
    with open(policy_path, "r") as f:
        return f.read()


def load_tasks(domain: str) -> List[Dict[str, Any]]:
    """
    Load tasks from file.

    Args:
        domain: Domain name (must exist in DOMAIN_PATHS)

    Returns:
        List of task dictionaries
    """
    tasks_path = DOMAIN_PATHS[domain]["tasks"]
    with open(tasks_path, "r") as f:
        return json.load(f)


def dict_to_task(task_dict: Dict[str, Any]) -> Task:
    """
    Convert a task dictionary to a tau2 Task pydantic model.
    
    Args:
        task_dict: Dictionary containing task data with keys like
                   id, description, user_scenario, initial_state, evaluation_criteria
    
    Returns:
        A tau2 Task model instance
    """
    return Task.model_validate(task_dict)
