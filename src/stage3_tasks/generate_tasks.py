#!/usr/bin/env python3
"""Stage 3 (generate): build validated tasks from a clusters artifact.

Loads validated cluster medoids produced by stage 2, generates a task per
usable cluster (one LLM call per task with patch-based retries), then
re-clusters and retries any clusters that failed end-to-end. Successful
tasks are checkpointed to ``task_sets/<name>/tasks.json`` after every
success so a crash or interrupt does not lose work; ``--resume`` skips any
cluster_idx already recorded as a success.

Example:
    python -m src.stage3_tasks.generate_tasks \\
        --clusters artifacts/validated_clusters_airline_k20.json \\
        --domain airline
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Path setup — must precede ``tau2`` imports.  Absolute so the script works
# regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tau2-bench" / "src"))

from src.common.domain_config import DomainConfig
from src.stage3_tasks.task_generation import TaskGenerator
from src.stage3_tasks.task_validator import TaskValidator
from src.stage3_tasks.task_builder import GeneratedTask
from src.stage3_tasks.task_set import TaskSetManager, TaskSetConfig
from src.common.sampler.length_distribution import LengthDistributionConfig
from src.common.sampler.action_sequence_clusterer import load_validated_clusters
from src.stage3_tasks.validation_with_retry import validate_task_with_retries


def _load_completed_cluster_idxs(progress_path: str) -> set:
    """Return the set of cluster_idxs that have at least one success record in progress.jsonl."""
    completed = set()
    if not os.path.exists(progress_path):
        return completed
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("success") and rec.get("cluster_idx") is not None:
                    completed.add(int(rec["cluster_idx"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


def _load_existing_tasks(task_set_dir: str) -> List[Dict]:
    """Load existing tasks.json if it exists, return empty list otherwise."""
    tasks_path = os.path.join(task_set_dir, "tasks.json")
    if not os.path.exists(tasks_path):
        return []
    with open(tasks_path) as f:
        return json.load(f)


def _checkpoint_save_tasks(task_set_dir: str, task_dicts: List[Dict]) -> None:
    """Write tasks to tasks.json immediately so resume can recover them after a crash."""
    tasks_path = os.path.join(task_set_dir, "tasks.json")
    cleaned = [TaskSetManager._clean_task_dict(t) for t in task_dicts]
    with open(tasks_path, "w") as f:
        json.dump(cleaned, f, indent=4)


def _short_model_name(model_name: str) -> str:
    """Extract a short name from a full model identifier.

    Example: ``vertex_ai/gemini-3-flash-preview`` -> ``gemini_3_flash``.
    """
    base = model_name.rsplit("/", 1)[-1]
    base = re.sub(r"-(preview|latest)$", "", base)
    return base.replace("-", "_")


def _build_entries_from_clusters(
    clusters_path: str,
    indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Load validated clusters and return usable entries with action sequences."""
    data = load_validated_clusters(clusters_path)
    clusters = data["clusters"]

    entries = []
    for cluster in clusters:
        if cluster.get("status") == "unusable":
            continue
        idx = cluster["cluster_idx"]
        if indices is not None and idx not in indices:
            continue
        entries.append({
            "action_sequence": cluster["representative_sequence"],
            "source": "medoid",
            "cluster_idx": idx,
        })

    return entries


def generate_single_task(
    entry: Dict[str, Any],
    index: int,
    n_total: int,
    domain: str,
    model_name: str,
    gt_llm: str,
    max_task_retries: int,
    gt_coverage_p: float = 0.33,
    gt_coverage_shuffle: bool = True,
    domain_config: Optional[DomainConfig] = None,
    max_patch_attempts: int = 3,
    write_only: bool = False,
) -> Tuple[Optional[GeneratedTask], Dict[str, Any]]:
    """Generate and validate a single task from an action sequence."""
    action_sequence = entry["action_sequence"]
    label = f"[{index + 1}/{n_total}]"
    print(f"{label} Generating task for sequence of length {len(action_sequence)}: "
          f"{action_sequence}")

    if domain_config is None:
        domain_config = DomainConfig(domain)

    generator = TaskGenerator(domain=domain, model_name=model_name, domain_config=domain_config)
    validator = TaskValidator(
        domain=domain,
        gt_llm=gt_llm,
        partial_coverage_p=gt_coverage_p,
        gt_coverage_shuffle=gt_coverage_shuffle,
        domain_config=domain_config,
        write_only=write_only,
    )

    t0 = time.time()

    def on_retry(retry_count, max_retries, db_only):
        kind = "DB-only" if db_only else "full"
        print(f"  {label} Retry {retry_count} ({kind})")

    success, task_dict, db_entities, result, retries_used = validate_task_with_retries(
        action_sequence=action_sequence,
        generator=generator,
        validator=validator,
        max_retries=max_task_retries,
        max_patch_attempts=max_patch_attempts,
        on_retry=on_retry,
    )

    elapsed = time.time() - t0

    meta = {
        "index": index,
        "source": entry.get("source"),
        "action_sequence": action_sequence,
        "success": success,
        "retries_used": retries_used,
        "elapsed_seconds": round(elapsed, 1),
    }
    if "cluster_idx" in entry:
        meta["cluster_idx"] = entry["cluster_idx"]
    if not success:
        meta["error"] = result.error

    if success:
        generated = GeneratedTask.create(task_dict, db_entities, domain=domain)
        generated.task.id = f"generated_{index}"
        generated.validation.success = True
        generated.validation.solver_retries = retries_used
        print(f"  {label} Success (retries={retries_used}, {elapsed:.1f}s)")
        return generated, meta
    else:
        print(f"  {label} Failed after {retries_used} retries ({elapsed:.1f}s): {result.error}")
        return None, meta


def _recluster_and_retry(
    failed_orig_indices: List[int],
    entries: List[Dict[str, Any]],
    all_generated: List,
    all_meta: List[Dict],
    clusters_path: str,
    domain: str,
    model_name: str,
    gt_llm: str,
    max_task_retries: int,
    progress_path: str,
    gt_coverage_p: float = 0.33,
    gt_coverage_shuffle: bool = True,
    recluster_pool_size: int = 500,
    recluster_temperature: float = 1.5,
    domain_config: Optional[DomainConfig] = None,
    on_task_success=None,
    max_patch_attempts: int = 3,
    write_only: bool = False,
) -> None:
    """Re-cluster using frozen valid medoids, then retry failed entries.

    Mutates ``all_generated`` and ``all_meta`` in place; appends new progress
    lines to ``progress_path``.
    """
    import random
    from src.common.sampler.action_sequence_ngram_sampler import ActionSequenceNGramModelSampler
    from src.common.sampler.action_sequence_clusterer import ActionSequenceClusterer
    from src.stage2_cluster.action_sequence_validator import ActionSequenceValidator as SeqValidator
    from src.common.tool_spec_retriever import ToolsSpecRetriever

    if domain_config is None:
        domain_config = DomainConfig(domain)

    data = load_validated_clusters(clusters_path)
    checkpoint_path = data.get("ngram_checkpoint")
    if not checkpoint_path:
        print("  WARNING: cluster artifact has no ngram_checkpoint, skipping re-cluster.")
        return

    print(f"  Loading n-gram model from {checkpoint_path}...")
    sampler = ActionSequenceNGramModelSampler.load_state(checkpoint_path)
    action_types = ToolsSpecRetriever(path=domain_config.tool_spec_path).get_action_types()

    failed_set = set(failed_orig_indices)
    frozen_seqs = [
        entries[i]["action_sequence"]
        for i in range(len(entries))
        if i not in failed_set and all_meta[i].get("success")
    ]
    n_new = len(failed_orig_indices)
    print(f"  Frozen: {len(frozen_seqs)} successful sequences, requesting {n_new} new clusters")

    if not frozen_seqs:
        print("  WARNING: no successful sequences to use as frozen centers; skipping re-cluster.")
        return

    clusterer = ActionSequenceClusterer(
        model=sampler.model,
        length_distribution=sampler.length_distribution,
        length_config=sampler._length_config,
        n_clusters=len(entries),
        pool_size=recluster_pool_size,
        temperature=recluster_temperature,
        action_types=action_types,
        rng=random.Random(42),
    )

    print(f"  Generating pool of {recluster_pool_size} sequences for re-clustering...")
    new_pool = clusterer.generate_pool()
    print(f"  Running recluster_with_frozen...")
    new_cluster_infos = clusterer.recluster_with_frozen(frozen_seqs, n_new, new_pool)

    # Validate new sequences: pick the first LLM-valid sequence per new cluster.
    seq_validator = SeqValidator(
        domain=domain,
        model_name=model_name,
        domain_config=domain_config,
    )
    new_sequences: List[Optional[List[str]]] = []
    for ci in new_cluster_infos:
        chosen = None
        for seq in ci.member_sequences:
            r = seq_validator.validate(seq)
            if r.valid:
                chosen = seq
                break
        if chosen is None:
            chosen = ci.medoid_sequence  # fallback: use the medoid even if not LLM-validated
        new_sequences.append(chosen)
        print(f"    New cluster {ci.cluster_idx}: {chosen}")

    # Retry generation with the new sequences.
    n_total_retry = len(failed_orig_indices)
    for retry_i, (orig_idx, new_seq) in enumerate(zip(failed_orig_indices, new_sequences)):
        if new_seq is None:
            continue
        new_entry: Dict[str, Any] = {
            "action_sequence": new_seq,
            "source": "reclustered",
        }
        print(f"  [recluster {retry_i + 1}/{n_total_retry}] Retrying orig_idx={orig_idx} "
              f"with new sequence: {new_seq}")
        generated, meta = generate_single_task(
            new_entry, orig_idx, n_total_retry, domain, model_name,
            gt_llm, max_task_retries,
            gt_coverage_p=gt_coverage_p,
            gt_coverage_shuffle=gt_coverage_shuffle,
            domain_config=domain_config,
            max_patch_attempts=max_patch_attempts,
            write_only=write_only,
        )
        meta["reclustered"] = True
        all_generated[orig_idx] = generated
        all_meta[orig_idx] = meta
        if generated is not None and on_task_success is not None:
            on_task_success(generated)
        with open(progress_path, "a") as f:
            f.write(json.dumps(meta) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate validated tasks from a stage-2 clusters artifact.",
    )
    parser.add_argument(
        "--clusters", type=str, required=True,
        help="Path to validated_clusters.json (output of stage 2).",
    )
    parser.add_argument("--indices", type=str, default=None,
                        help="Comma-separated cluster indices to use (default: all usable).")
    parser.add_argument("--domain", type=str, default="airline",
                        help="Domain name (default: airline).")
    parser.add_argument("--model-name", type=str, default="gemini/gemini-3-flash-preview",
                        help="LLM model for task generation.")
    parser.add_argument("--gt-llm", type=str, default="gemini/gemini-3-flash-preview",
                        help="LLM model for GT agent validation.")
    parser.add_argument("--gt-coverage-p", type=float, default=0.33,
                        help="Fraction of assistant actions to cover in partial_coverage mode (default: 0.33).")
    parser.add_argument("--gt-coverage-shuffle", action=argparse.BooleanOptionalAction, default=True,
                        help="Shuffle action order in partial_coverage mode (default: True). "
                             "Use --no-gt-coverage-shuffle to keep original order.")
    parser.add_argument("--max-task-retries", type=int, default=3,
                        help="Max retries per task (default: 3).")
    parser.add_argument("--name", type=str, default=None,
                        help="Task set name (default: auto-generated).")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        metavar="TASK_SET_NAME",
                        help="Resume a previous run. Pass a task set name (e.g. 'airline_gemini_3_flash_easy') "
                             "or omit the value to auto-detect from --domain/--model-name. "
                             "Skips succeeded entries in progress.jsonl and appends new results.")
    parser.add_argument(
        "--write-only",
        action="store_true",
        default=False,
        help="Telecom only. Signal to the GT agent that the listed action list "
             "contains only WRITE actions; the agent is told reads (customer "
             "lookups, device/network checks, bill queries) are unlisted but "
             "may be called freely. Auto-enabled when the clusters file has "
             "`_transform` containing 'strip_reads' in its metadata.",
    )
    parser.add_argument(
        "--max-patch-attempts",
        type=int,
        default=3,
        help="Max patch-LLM calls per failed validation step before falling back "
             "to full regeneration. Each patch is ~30s with GPT-5.2. Set to 0 to "
             "skip patching entirely (fastest; recommended for telecom where many "
             "medoid sequences can't be coherently authored and patches just "
             "burn LLM calls re-hitting the same wall). Default 3.",
    )

    args = parser.parse_args()

    domain_config = DomainConfig(args.domain)

    indices = None
    if args.indices is not None:
        indices = [int(x.strip()) for x in args.indices.split(",")]

    print(f"Loading validated clusters from '{args.clusters}'...")
    entries = _build_entries_from_clusters(args.clusters, indices)

    if not entries:
        print("No action sequences to process. Exiting.")
        sys.exit(1)

    # Auto-detect write-only medoid files (produced by the read-stripping
    # transform). These files declare `_transform` in their top-level
    # metadata and contain no customer-lookup reads; the downstream task
    # generator supplies IDs via known_info, and the GT agent should be
    # told the list is writes-only and that it may call reads at will.
    if args.domain == "telecom" and args.clusters and os.path.exists(args.clusters):
        try:
            with open(args.clusters) as f:
                _clusters_meta = json.load(f)
            transform = _clusters_meta.get("_transform") or ""
            if "strip_reads" in transform and not args.write_only:
                args.write_only = True
                print(
                    f"  Detected write-only medoid file "
                    f"(_transform='{transform}'); "
                    f"enabling --write-only (GT agent will be told reads are unlisted "
                    f"but may be called freely)."
                )
        except Exception:
            pass

    print(f"Found {len(entries)} action sequences to generate tasks for.")

    short_model = _short_model_name(args.model_name)
    if args.resume and args.resume != "auto":
        name = args.resume
    else:
        name = args.name or f"{args.domain}_{short_model}"
    print(f"Task set name: {name}")

    task_set_dir = os.path.join("task_sets", name)
    os.makedirs(task_set_dir, exist_ok=True)
    progress_path = os.path.join(task_set_dir, "progress.jsonl")

    # Resume: filter out cluster_idxs already recorded as a success.
    existing_tasks: List[Dict] = []
    completed_cluster_idxs: set = set()
    if args.resume is not None:
        if not os.path.exists(progress_path):
            print(f"WARNING: No progress.jsonl found at {progress_path}, starting fresh.")
        completed_cluster_idxs = _load_completed_cluster_idxs(progress_path)
        existing_tasks = _load_existing_tasks(task_set_dir)

        if completed_cluster_idxs:
            print(f"Resuming: {len(completed_cluster_idxs)} cluster_idxs already succeeded, "
                  f"{len(existing_tasks)} existing tasks in tasks.json")
            pending = [(i, e) for i, e in enumerate(entries)
                       if e.get("cluster_idx") not in completed_cluster_idxs]
            print(f"  {len(pending)} entries to retry")
        else:
            pending = list(enumerate(entries))
    else:
        pending = list(enumerate(entries))

    if not pending:
        print("All entries already completed. Nothing to do.")
        sys.exit(0)

    all_generated: List[Optional[GeneratedTask]] = [None] * len(entries)
    all_meta: List[Dict[str, Any]] = [{}] * len(entries)

    # Running list of completed task dicts (existing + newly succeeded).
    # Checkpointed to tasks.json after each success so a crash doesn't lose work.
    checkpoint_tasks: List[Dict] = list(existing_tasks)

    def _on_task_success(generated: GeneratedTask) -> None:
        checkpoint_tasks.append(generated.to_dict())
        _checkpoint_save_tasks(task_set_dir, checkpoint_tasks)

    n_total = len(pending)

    for seq_idx, (orig_idx, entry) in enumerate(pending):
        generated, meta = generate_single_task(
            entry, orig_idx, n_total, args.domain, args.model_name,
            args.gt_llm, args.max_task_retries,
            gt_coverage_p=args.gt_coverage_p,
            gt_coverage_shuffle=args.gt_coverage_shuffle,
            domain_config=domain_config,
            max_patch_attempts=args.max_patch_attempts,
            write_only=args.write_only,
        )
        all_generated[orig_idx] = generated
        all_meta[orig_idx] = meta
        if generated is not None:
            _on_task_success(generated)
        with open(progress_path, "a") as f:
            f.write(json.dumps(meta) + "\n")

    # Re-cluster and retry any clusters that failed end-to-end.
    failed_orig_indices = [
        orig_idx
        for orig_idx, meta in enumerate(all_meta)
        if meta.get("index") is not None and not meta.get("success")
    ]
    if failed_orig_indices:
        print(f"\n--- Re-clustering on failure: {len(failed_orig_indices)} failed entries ---")
        _recluster_and_retry(
            failed_orig_indices=failed_orig_indices,
            entries=entries,
            all_generated=all_generated,
            all_meta=all_meta,
            clusters_path=args.clusters,
            domain=args.domain,
            model_name=args.model_name,
            gt_llm=args.gt_llm,
            max_task_retries=args.max_task_retries,
            progress_path=progress_path,
            gt_coverage_p=args.gt_coverage_p,
            gt_coverage_shuffle=args.gt_coverage_shuffle,
            domain_config=domain_config,
            on_task_success=_on_task_success,
            max_patch_attempts=args.max_patch_attempts,
            write_only=args.write_only,
        )
    else:
        print("  No failed entries; re-clustering skipped.")

    new_successful = [g for g in all_generated if g is not None]
    n_new_success = len(new_successful)
    n_new_failed = n_total - n_new_success
    print(f"\nThis run: {n_new_success}/{n_total} tasks generated successfully, {n_new_failed} failed.")

    # checkpoint_tasks already contains existing_tasks + all newly succeeded tasks.
    all_task_dicts = checkpoint_tasks
    if args.resume is not None and existing_tasks:
        print(f"Merged: {len(existing_tasks)} existing + {n_new_success} new = {len(all_task_dicts)} total")

    if not all_task_dicts:
        print("No tasks generated. Skipping save.")
        sys.exit(1)

    config = TaskSetConfig(
        name=name,
        domain=args.domain,
        seed_source="medoids",
        num_tasks_to_generate=len(entries),
        length_distribution_config=LengthDistributionConfig(mode="from_seed"),
        model_name=args.model_name,
    )

    manager = TaskSetManager()
    saved_path = manager.save(
        name=name,
        all_tasks=all_task_dicts,
        config=config,
        num_seed_tasks=0,
        num_generated_tasks=len(all_task_dicts),
    )
    print(f"Task set saved to: {saved_path}")

    generation_meta = {
        "name": name,
        "domain": args.domain,
        "model_name": args.model_name,
        "gt_llm": args.gt_llm,
        "gt_coverage_p": args.gt_coverage_p,
        "gt_coverage_shuffle": args.gt_coverage_shuffle,
        "seed_source": "medoids",
        "max_task_retries": args.max_task_retries,
        "total_sequences": len(entries),
        "successful_this_run": n_new_success,
        "failed_this_run": n_new_failed,
        "total_tasks_saved": len(all_task_dicts),
        "resumed": args.resume is not None,
        "created_at": datetime.now().isoformat(),
        "tasks": all_meta,
    }
    meta_path = os.path.join(task_set_dir, "generation_meta.json")
    with open(meta_path, "w") as f:
        json.dump(generation_meta, f, indent=2)
    print(f"Generation metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
