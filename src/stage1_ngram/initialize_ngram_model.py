#!/usr/bin/env python3
"""Stage 1: train an n-gram action sequence model.

Loads seed tasks for the chosen domain, validates them with an LLM, then runs
the sampler to generate, validate and ingest new sequences. Writes JSON
checkpoints into the run directory at four points (``pre_seed``, ``post_seed``,
periodic every ``--save-every`` samples, and ``final``), plus a descriptive
copy of the final state into ``artifacts/ngram/checkpoints/`` for downstream
stages to consume.

Example:
    python -m src.stage1_ngram.initialize_ngram_model \\
        --domain airline --num-samples 500
"""

import argparse
import os

from src.common.domain_utils import WORKSPACE_ROOT
from src.common.sampler import LengthDistributionConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an n-gram action sequence model with periodic checkpoints",
    )
    parser.add_argument("--domain", type=str, default="airline", help="Task domain (default: airline)")
    parser.add_argument(
        "--seed-tasks", type=str, default=None,
        help="Path to a tasks.json file to use as seeds instead of the default domain tasks (e.g. task_sets/verified_airline_tasks.json)",
    )
    parser.add_argument("--num-samples", type=int, default=3000, help="Total samples to run (default: 3000)")
    parser.add_argument("--save-every", type=int, default=400, help="Checkpoint interval (default: 400)")
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help=(
            "Directory for checkpoints. Defaults to artifacts/ngram/training/<domain>/, "
            "with airline_no_neg used as the subdir when --no-ingest-negatives is set on airline."
        ),
    )
    parser.add_argument(
        "--validation-model",
        default="gemini/gemini-3-flash-preview",
        help="LLM model for validation",
    )
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="Max tokens for validator LLM response")

    ngram = parser.add_argument_group("NGram model parameters")
    ngram.add_argument("--n", type=int, default=3, help="N-gram order (default: 3)")
    ngram.add_argument("--alpha0", type=float, default=0.1, help="Dirichlet prior (default: 0.1)")
    ngram.add_argument("--lambda-neg", type=float, default=1.0, help="Negative penalty strength (default: 1.0)")
    ngram.add_argument("--tau0", type=float, default=3.0, help="Initial temperature (default: 2.0)")
    ngram.add_argument("--tau-decay-steps", type=float, default=1500.0, help="Temperature decay steps (default: 1500)")
    ngram.add_argument("--negative-weight", type=float, default=1.0, help="Weight for negative ingestion (default: 1.0)")
    ngram.add_argument("--no-ingest-negatives", action="store_true", help="Do not ingest invalid sequences as negatives")

    return parser.parse_args()


def main():
    args = parse_args()

    from src.common.domain_config import DomainConfig
    from src.common.sampler import ActionSequenceNGramModelSampler
    from src.stage2_cluster.action_sequence_validator import ActionSequenceValidator

    # Use a single canonical run dir per training type; re-running overwrites
    # the previous run for that type.
    if args.save_dir is None:
        sub = args.domain
        if args.domain == "airline" and args.no_ingest_negatives:
            sub = "airline_no_neg"
        args.save_dir = os.path.join("artifacts", "ngram", "training", sub)
    run_dir = args.save_dir
    os.makedirs(run_dir, exist_ok=True)

    domain = args.domain
    domain_config = DomainConfig(domain)

    import json
    from tau2.data_model.tasks import Task
    if args.seed_tasks:
        print(f"\nLoading seed tasks from file: {args.seed_tasks}")
        with open(args.seed_tasks) as f:
            seed_tasks = [Task.model_validate(t) for t in json.load(f)]
    else:
        from src.common.domain_utils import load_tasks as load_domain_tasks_raw
        print(f"\nLoading seed tasks for domain '{args.domain}'...")
        raw_tasks = load_domain_tasks_raw(args.domain)
        seed_tasks = [Task.model_validate(t) for t in raw_tasks]
    print(f"Loaded {len(seed_tasks)} seed tasks")

    sampler = ActionSequenceNGramModelSampler(
        action_sequence_set=seed_tasks,
        length_distribution_config=LengthDistributionConfig(
            mode="from_seed",
            reference_source="original",
        ),
        domain=args.domain,
        n=args.n,
        alpha0=args.alpha0,
        lambda_neg=args.lambda_neg,
        tau0=args.tau0,
        tau_decay_steps=args.tau_decay_steps,
        negative_weight=args.negative_weight,
        ingest_negatives=not args.no_ingest_negatives,
        tool_spec_path=domain_config.tool_spec_path,
    )

    validator = ActionSequenceValidator(
        domain=domain,
        model_name=args.validation_model,
        max_output_tokens=args.max_output_tokens,
        domain_config=domain_config,
    )

    pre_seed_path = os.path.join(run_dir, "pre_seed.json")
    sampler.save_state(pre_seed_path)
    print(f"  >>> Checkpoint saved: {pre_seed_path}  [pre_seed]")

    pending = sampler._pending_seeds
    print(f"\nValidating {len(pending)} seed action sequences...")
    valid = invalid = skipped = 0
    for i, seq in enumerate(pending):
        if not seq:
            skipped += 1
            continue
        result = validator.validate(seq)
        if result.valid:
            sampler.model.ingest_sequence(seq, weight=sampler._positive_weight, positive=True)
            sampler.accepted_set.add(tuple(seq))
            sampler.action_sequence_set.append(seq)
            valid += 1
        else:
            invalid += 1
            if sampler._ingest_negatives and result.problematic_indices:
                sampler.model.ingest_negative_windows_at_indices(
                    seq, result.problematic_indices, weight=sampler._negative_weight,
                )
        if (i + 1) % max(1, len(pending) // 5) == 0:
            print(f"  Validated [{i+1}/{len(pending)}] Valid: {valid}  Invalid: {invalid}  Skipped: {skipped}")
    print(f"\n  Valid: {valid}  Invalid: {invalid}  Skipped: {skipped}")

    post_seed_path = os.path.join(run_dir, "post_seed.json")
    sampler.save_state(post_seed_path)
    print(f"  >>> Checkpoint saved: {post_seed_path}  [post_seed]")

    print(f"\nTraining for {args.num_samples} samples, saving every {args.save_every}")
    print(f"  Validation model: {args.validation_model}")
    print(f"  n={args.n}  alpha0={args.alpha0}  lambda_neg={args.lambda_neg}  tau0={args.tau0}")
    print()

    accepted_total = 0
    rejected_total = 0

    # Per-window counters (reset every save_every).
    w_accepted = 0
    w_rejected = 0
    window_start = 1
    window_stats: list = []

    for i in range(1, args.num_samples + 1):
        action_sequence, _ = sampler.get_new_action_sequence()
        tau = sampler.current_temperature

        result = validator.validate(action_sequence)
        sampler.ingest(
            action_sequence,
            positive=result.valid,
            problematic_indices=result.problematic_indices if not result.valid else None,
        )

        if result.valid:
            accepted_total += 1
            w_accepted += 1
        else:
            rejected_total += 1
            w_rejected += 1

        label = "VALID" if result.valid else "INVLD"
        print(f"  [{sampler.t:>5}/{args.num_samples}]  {label}  tau={tau:.3f}  accepted={accepted_total}  len={len(action_sequence)}")

        if i % args.save_every == 0:
            w_attempted = w_accepted + w_rejected
            window_stats.append({
                "window_index": len(window_stats),
                "sample_start": window_start,
                "sample_end": i,
                "attempted": w_attempted,
                "accepted": w_accepted,
                "rejected": w_rejected,
                "acceptance_rate": w_accepted / w_attempted if w_attempted > 0 else None,
            })

            filename = sampler.checkpoint_filename()
            path = os.path.join(run_dir, filename)
            sampler.save_state(path)
            print(f"  >>> Checkpoint saved: {path}  [periodic]")

            w_accepted = 0
            w_rejected = 0
            window_start = i + 1

    # Capture any partial window that didn't hit save_every.
    if w_accepted + w_rejected > 0:
        w_attempted = w_accepted + w_rejected
        window_stats.append({
            "window_index": len(window_stats),
            "sample_start": window_start,
            "sample_end": args.num_samples,
            "attempted": w_attempted,
            "accepted": w_accepted,
            "rejected": w_rejected,
            "acceptance_rate": w_accepted / w_attempted if w_attempted > 0 else None,
        })

    filename = sampler.checkpoint_filename()
    name, ext = os.path.splitext(filename)
    final_filename = f"{name}_final{ext}"
    path = os.path.join(run_dir, final_filename)
    sampler.save_state(path)
    print(f"  >>> Checkpoint saved: {path}  [final]")

    # Also save into artifacts/ngram/checkpoints/ with a descriptive name as
    # the convenience handle for src.stage2_cluster.cluster_and_validate.
    no_neg_suffix = "_no_neg" if (domain == "airline" and args.no_ingest_negatives) else ""
    checkpoints_dir = os.path.join(WORKSPACE_ROOT, "artifacts", "ngram", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    artifact_name = f"ngram_checkpoint_{domain}{no_neg_suffix}_{args.num_samples}samples.json"
    artifact_path = os.path.join(checkpoints_dir, artifact_name)
    sampler.save_state(artifact_path)
    print(f"  >>> Checkpoint saved: {artifact_path}  [artifact]")

    print(f"\nFinal state saved:  {path}")
    print(f"  Artifact:         {artifact_path}")
    print(f"  Target accepted:  {args.num_samples}")
    print(f"  Accepted (total): {len(sampler.accepted_set)}")
    print(f"  Accepted (run):   {accepted_total}")
    print(f"  Rejected:         {rejected_total}")
    overall_attempted = accepted_total + rejected_total
    if overall_attempted > 0:
        print(f"  Overall acc rate: {accepted_total / overall_attempted:.1%}")
    print(f"  Final t:          {sampler.t}")
    print()
    if window_stats:
        print("  Per-window acceptance rates:")
        for w in window_stats:
            rate = f"{w['acceptance_rate']:.1%}" if w['acceptance_rate'] is not None else "n/a"
            print(f"    [{w['sample_start']:>5}–{w['sample_end']:<5}]  accepted={w['accepted']}  rejected={w['rejected']}  rate={rate}")
    print(f"\n  Next step: python -m src.stage2_cluster.cluster_and_validate --checkpoint {artifact_path} -k <num_clusters>")
    print()


if __name__ == "__main__":
    main()
