#!/usr/bin/env python3
"""
Adversarial task evolution: 3-phase pipeline to make tasks genuinely hard.

Phase 1: Adversarial strategy (analyze golden actions + DB)
Phase 2: DB trap construction (build decoy entities)
Phase 3: Scenario writing (adversarial user instructions)

Validates with GT agent only. Falls back to original if all retries fail.

Usage:
  python -m src.stage3_tasks.evolve_tasks --task-set task_sets/airline_v1_medoids_easy
  python -m src.stage3_tasks.evolve_tasks --task-set task_sets/airline_v1_medoids_easy \\
      --output-name airline_adversarial_v1
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup — must precede ``tau2`` imports.  Absolute so the script works
# regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tau2-bench" / "src"))

from src.stage3_tasks.adversarial_evolver import AdversarialEvolver
from src.common.domain_config import DomainConfig
from src.stage3_tasks.task_validator import TaskValidator
from src.common.domain_utils import WORKSPACE_ROOT

TASK_SETS_DIR = os.path.join(WORKSPACE_ROOT, "task_sets")


def _load_tasks(task_set: str) -> List[Dict[str, Any]]:
    """Load tasks.json from a task set directory or direct file path."""
    if os.path.isfile(task_set):
        path = task_set
    elif os.path.isfile(os.path.join(task_set, "tasks.json")):
        path = os.path.join(task_set, "tasks.json")
    else:
        candidate = os.path.join(TASK_SETS_DIR, task_set, "tasks.json")
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Cannot find tasks.json for task set '{task_set}'")
        path = candidate
    with open(path) as f:
        return json.load(f)


def _assemble_task(
    original: Dict[str, Any],
    scenario: Dict[str, str],
    db_additions: Dict[str, Any],
) -> Dict[str, Any]:
    """Deep-copy original task, patch scenario + merge DB additions."""
    task = copy.deepcopy(original)

    # Patch user scenario instructions
    instr = task.setdefault("user_scenario", {}).setdefault("instructions", {})
    instr["reason_for_call"] = scenario.get("reason_for_call", instr.get("reason_for_call", ""))
    instr["known_info"] = scenario.get("known_info", instr.get("known_info", ""))
    instr["task_instructions"] = scenario.get("task_instructions", instr.get("task_instructions", ""))

    # Merge DB additions into initial_state
    agent_data = (
        task.setdefault("initial_state", {})
        .setdefault("initialization_data", {})
        .setdefault("agent_data", {})
    )

    # Add decoy flights (airline)
    for fnum, flight in db_additions.get("flights", {}).items():
        agent_data.setdefault("flights", {})[fnum] = flight

    # Add decoy reservations (airline)
    for rid, res in db_additions.get("reservations", {}).items():
        agent_data.setdefault("reservations", {})[rid] = res
        res_user_id = res.get("user_id", "")
        for uid, user in agent_data.get("users", {}).items():
            if uid == res_user_id:
                if rid not in user.get("reservations", []):
                    user.setdefault("reservations", []).append(rid)

    # Add decoy orders (retail)
    for oid, order in db_additions.get("orders", {}).items():
        agent_data.setdefault("orders", {})[oid] = order
        order_user_id = order.get("user_id", "")
        for uid, user in agent_data.get("users", {}).items():
            if uid == order_user_id:
                if oid not in user.get("orders", []):
                    user.setdefault("orders", []).append(oid)

    # --- Telecom decoys (agent_data uses LIST-of-dicts, not dict-of-dicts) ---
    # Customers — append new customers to the list.
    for cust in db_additions.get("customers", []):
        agent_data.setdefault("customers", []).append(cust)

    # Plans — append new plans to the list.
    for plan in db_additions.get("plans", []):
        agent_data.setdefault("plans", []).append(plan)

    # Devices — append new devices to the list.
    for device in db_additions.get("devices", []):
        agent_data.setdefault("devices", []).append(device)

    # Lines — append, then link back to owning customer by customer_id.
    for line in db_additions.get("lines", []):
        agent_data.setdefault("lines", []).append(line)
        line_cid = line.get("customer_id", "")
        lid = line.get("line_id", "")
        if not line_cid or not lid:
            continue
        for cust in agent_data.get("customers", []):
            if cust.get("customer_id") == line_cid:
                if lid not in cust.setdefault("line_ids", []):
                    cust["line_ids"].append(lid)
                break

    # Bills — append, then link back to owning customer by customer_id.
    for bill in db_additions.get("bills", []):
        agent_data.setdefault("bills", []).append(bill)
        bill_cid = bill.get("customer_id", "")
        bid = bill.get("bill_id", "")
        if not bill_cid or not bid:
            continue
        for cust in agent_data.get("customers", []):
            if cust.get("customer_id") == bill_cid:
                if bid not in cust.setdefault("bill_ids", []):
                    cust["bill_ids"].append(bid)
                break

    return task


def _evolve_one(
    task: Dict[str, Any],
    model_name: str,
    gt_llm: str,
    domain: str,
    max_phase1_retries: int = 2,
    max_phase3_retries: int = 3,
    domain_config: Optional[DomainConfig] = None,
) -> Tuple[Dict[str, Any], bool, Optional[str], Optional[Dict]]:
    """
    Evolve a single task with nested retry loop.
    Returns (result_task, evolved_flag, error_or_None, strategy_or_None).
    """
    evolver = AdversarialEvolver(model_name=model_name, domain=domain, domain_config=domain_config)
    validator = TaskValidator(
        domain=domain,
        gt_llm=gt_llm,
        adversarial_user=True,
        domain_config=domain_config,
    )

    last_error = None
    last_strategy = None

    for p1_attempt in range(1, max_phase1_retries + 1):
        # Phase 1: Strategy
        try:
            strategy = evolver.phase1_strategy(task)
            last_strategy = strategy
        except Exception as e:
            last_error = f"Phase 1 error (attempt {p1_attempt}): {e}"
            print(f"    {last_error}")
            continue

        # Phase 2: DB traps
        try:
            db_additions = evolver.phase2_db_traps(strategy, task)
        except Exception as e:
            last_error = f"Phase 2 error (attempt {p1_attempt}): {e}"
            print(f"    {last_error}")
            continue

        # Phase 3: Scenario (with inner retry loop)
        for p3_attempt in range(1, max_phase3_retries + 1):
            try:
                scenario = evolver.phase3_scenario(strategy, db_additions, task)
            except Exception as e:
                last_error = f"Phase 3 error (p1={p1_attempt}, p3={p3_attempt}): {e}"
                print(f"    {last_error}")
                continue

            evolved_task = _assemble_task(task, scenario, db_additions)

            try:
                result = validator.validate_gt_only(evolved_task, verify_env_assertions=True)
            except Exception as e:
                last_error = f"GT validation crashed (p1={p1_attempt}, p3={p3_attempt}): {e}"
                print(f"    {last_error}")
                continue

            if result.gt_agent_success:
                evolved_task["evolved"] = True
                total_attempts = (p1_attempt - 1) * max_phase3_retries + p3_attempt
                if total_attempts > 1:
                    print(f"    (succeeded on p1={p1_attempt}, p3={p3_attempt})")
                return evolved_task, True, None, strategy
            else:
                last_error = f"GT failed (p1={p1_attempt}, p3={p3_attempt}): {result.error}"
                print(f"    {last_error}")

    # Lite fallback: reduced adversarial treatment (no strategy/traps, just mild elements)
    print(f"    Full adversarial failed, trying lite version...")
    for lite_attempt in range(1, max_phase3_retries + 1):
        try:
            scenario = evolver.phase3_scenario_lite(task)
        except Exception as e:
            last_error = f"Lite Phase 3 error (attempt {lite_attempt}): {e}"
            print(f"    {last_error}")
            continue

        evolved_task = _assemble_task(task, scenario, {})

        try:
            result = validator.validate_gt_only(evolved_task, verify_env_assertions=True)
        except Exception as e:
            last_error = f"Lite GT validation crashed (attempt {lite_attempt}): {e}"
            print(f"    {last_error}")
            continue

        if result.gt_agent_success:
            evolved_task["evolved"] = True
            evolved_task["evolved_lite"] = True
            print(f"    (lite succeeded on attempt {lite_attempt})")
            return evolved_task, True, None, {"lite": True}
        else:
            last_error = f"Lite GT failed (attempt {lite_attempt}): {result.error}"
            print(f"    {last_error}")

    # Final fallback to original
    fallback = copy.deepcopy(task)
    fallback["evolved"] = False
    return fallback, False, last_error, last_strategy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial task evolution: 3-phase pipeline."
    )
    parser.add_argument(
        "--task-set", required=True,
        help="Path to task set directory (or name under task_sets/) to evolve.",
    )
    parser.add_argument(
        "--output-name",
        help="Output directory name under task_sets/. Defaults to <input>_adversarial.",
    )
    parser.add_argument("--domain", default="airline")
    parser.add_argument(
        "--model-name", default="vertex_ai/gemini-3-flash-preview",
        help="LLM for all 3 evolution phases.",
    )
    parser.add_argument(
        "--gt-llm", default="vertex_ai/gemini-3-flash-preview",
        help="LLM for GT agent validation.",
    )
    parser.add_argument("--max-phase1-retries", type=int, default=3)
    parser.add_argument("--max-phase3-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--indices", type=str, default=None,
        help="Comma-separated task indices (0-based, into the input tasks.json) to evolve. Default: all.",
    )
    args = parser.parse_args()

    # Resolve output directory
    input_name = os.path.basename(args.task_set.rstrip("/"))
    output_name = args.output_name or f"{input_name}_adversarial"
    output_dir = os.path.join(TASK_SETS_DIR, output_name)

    # Load input tasks
    tasks = _load_tasks(args.task_set)
    print(f"Loaded {len(tasks)} tasks from '{args.task_set}'")

    # Optional index filter (0-based positions in the loaded list)
    if args.indices is not None:
        try:
            wanted = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        except ValueError as e:
            print(f"Error: --indices must be comma-separated integers (got '{args.indices}'): {e}")
            sys.exit(1)
        out_of_range = [i for i in wanted if i < 0 or i >= len(tasks)]
        if out_of_range:
            print(f"Error: --indices {out_of_range} out of range for {len(tasks)} tasks.")
            sys.exit(1)
        tasks = [tasks[i] for i in wanted]
        print(f"Filtered to {len(tasks)} task(s) via --indices {args.indices}")

    # Handle resume — auto-detect existing output
    already_done: set = set()
    existing_results: Dict[str, Dict[str, Any]] = {}
    done_path = os.path.join(output_dir, "tasks.json")

    if os.path.isfile(done_path):
        try:
            with open(done_path) as f:
                existing_tasks = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\nError: output file '{done_path}' has invalid JSON (likely truncated from a previous crash).")
            print(f"  {e}")
            print(f"\nTo fix: delete or move the corrupted file, then re-run:")
            print(f"  rm '{done_path}'")
            sys.exit(1)

        n_evolved = sum(1 for t in existing_tasks if t.get("evolved") is True)
        n_fallback = sum(1 for t in existing_tasks if t.get("evolved") is False)

        if not args.resume:
            print(f"\nOutput '{output_name}' already exists: {n_evolved} evolved, {n_fallback} fallback, {len(existing_tasks)} total.")
            answer = input("Resume and process remaining tasks? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("Aborting. Use --output-name for a different output.")
                sys.exit(0)

        for t in existing_tasks:
            existing_results[t["id"]] = t
            if t.get("evolved") is True:
                already_done.add(t["id"])
        print(f"Resuming: {len(already_done)} tasks already evolved, skipping.")

    os.makedirs(output_dir, exist_ok=True)

    to_process = [t for t in tasks if t["id"] not in already_done]
    print(f"Processing {len(to_process)} tasks...")

    domain_config = DomainConfig(args.domain)

    results: Dict[str, Dict[str, Any]] = dict(existing_results)
    strategies: Dict[str, Optional[Dict]] = {}

    for i, task in enumerate(to_process):
        print(f"\n[{i + 1}/{len(to_process)}] Task {task['id']}")
        result_task, evolved, error, strategy = _evolve_one(
            task,
            args.model_name,
            args.gt_llm,
            args.domain,
            args.max_phase1_retries,
            args.max_phase3_retries,
            domain_config=domain_config,
        )
        results[task["id"]] = result_task
        strategies[task["id"]] = strategy
        status = "Evolved" if evolved else f"Kept original ({error})"
        print(f"  -> {status}")

        # Save progress after each task.
        _save_output(output_dir, tasks, results, strategies, args)

    # Print summary
    ordered = [results[t["id"]] for t in tasks if t["id"] in results]
    n_evolved = sum(1 for t in ordered if t.get("evolved") is True)
    n_fallback = sum(1 for t in ordered if t.get("evolved") is False)
    print(f"\n{'='*50}")
    print(f"Done: {n_evolved}/{len(ordered)} evolved, {n_fallback} kept original")
    print(f"Saved to {output_dir}")


def _save_output(
    output_dir: str,
    all_tasks: List[Dict],
    results: Dict[str, Dict],
    strategies: Dict[str, Optional[Dict]],
    args: argparse.Namespace,
) -> None:
    """Write tasks.json, metadata.json, and strategies.jsonl."""
    ordered = [results[t["id"]] for t in all_tasks if t["id"] in results]

    # tasks.json
    with open(os.path.join(output_dir, "tasks.json"), "w") as f:
        json.dump(ordered, f, indent=4)

    # metadata.json
    n_evolved = sum(1 for t in ordered if t.get("evolved") is True)
    n_fallback = sum(1 for t in ordered if t.get("evolved") is False)
    metadata = {
        "source_task_set": args.task_set,
        "output_name": args.output_name or f"{os.path.basename(args.task_set.rstrip('/'))}_adversarial",
        "created_at": datetime.now().isoformat(),
        "pipeline": "adversarial_evolution_v1",
        "num_tasks": len(ordered),
        "num_evolved": n_evolved,
        "num_fallback": n_fallback,
        "model_name": args.model_name,
        "gt_llm": args.gt_llm,
        "max_phase1_retries": args.max_phase1_retries,
        "max_phase3_retries": args.max_phase3_retries,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    # strategies.jsonl — one line per task for debugging
    with open(os.path.join(output_dir, "strategies.jsonl"), "w") as f:
        for t in all_tasks:
            tid = t["id"]
            if tid in strategies and strategies[tid] is not None:
                entry = {"task_id": tid, "strategy": strategies[tid]}
                f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    main()
