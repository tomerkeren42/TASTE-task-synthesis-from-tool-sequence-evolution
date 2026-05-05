#!/usr/bin/env python3
"""Stage 2: cluster generated action sequences and validate the medoids.

Loads a trained n-gram checkpoint, generates a pool of sequences, clusters
them via k-medoids, then validates each cluster's medoid with an LLM. Invalid
medoids are replaced from their cluster members; clusters that remain unusable
are re-clustered up to ``--max-recluster-rounds`` times. Writes the validated
clusters as a JSON artifact for stage 3 to consume.

Example:
    python -m src.stage2_cluster.cluster_and_validate \\
        --checkpoint artifacts/ngram/checkpoints/ngram_checkpoint_airline_3000samples.json \\
        -k 10 --pool-size 1000 --domain airline
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# Path setup — must precede ``tau2`` imports.  Absolute so the script works
# regardless of the caller's CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "tau2-bench" / "src"))

from src.stage2_cluster.action_sequence_validator import ActionSequenceValidator
from src.stage2_cluster.medoid_validation import validate_clusters
from src.common.domain_config import DomainConfig
from src.common.sampler.action_sequence_clusterer import (
    ActionSequenceClusterer,
    ClusterInfo,
    SubstitutionCosts,
    save_validated_clusters,
)
from src.common.sampler.action_sequence_ngram_sampler import ActionSequenceNGramModelSampler
from src.common.sampler.length_distribution import build_length_distribution
from src.common.tool_spec_retriever import ToolsSpecRetriever


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster action sequences and validate medoids (stage 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained n-gram checkpoint (file or directory)",
    )
    parser.add_argument(
        "-k", "--num-clusters", type=int, required=True,
        help="Number of clusters for k-medoids",
    )
    parser.add_argument(
        "--domain", default=None,
        help="Domain name (auto-detected from checkpoint; override with this flag)",
    )

    parser.add_argument("--pool-size", type=int, default=2000, help="Pool size for sequence generation")
    parser.add_argument("--pool-temperature", type=float, default=1.5, help="Temperature for pool generation")
    parser.add_argument("--max-kmedoids-iters", type=int, default=100, help="Max k-medoids iterations")

    parser.add_argument("--gaussian-mean", type=float, default=7, help="Mean (loc) of the skew-normal length distribution")
    parser.add_argument("--gaussian-std", type=float, default=5, help="Std (scale) of the skew-normal length distribution")
    parser.add_argument("--skew-alpha", type=float, default=2.0, help="Skewness (0=symmetric, >0=right-skewed, <0=left-skewed)")
    parser.add_argument("--min-length", type=int, default=1, help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=15, help="Maximum sequence length")

    parser.add_argument("--only-write", action="store_true",
                        help="Filter pool to sequences containing only WRITE/GENERIC actions (no READ).")
    parser.add_argument("--regular-edit-distance", action="store_true",
                        help="Use plain Levenshtein edit distance (all substitution costs = 1.0) "
                             "instead of the type/group-weighted variant.")

    parser.add_argument("--validation-model", default="gemini-3-flash-preview", help="LLM model for validation")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="Max output tokens for validation")
    parser.add_argument("--max-replacements", type=int, default=99, help="Max replacement candidates per invalid medoid")
    parser.add_argument(
        "--max-recluster-rounds", type=int, default=3,
        help="Number of re-clustering rounds to attempt for unusable clusters (default: 3).",
    )

    parser.add_argument(
        "--output", default=None,
        help="Output path for validated clusters artifact (default: auto-generated)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    t_main_start = time.time()

    sampler = ActionSequenceNGramModelSampler.load_state(args.checkpoint)

    # Auto-detect domain from checkpoint; --domain overrides.
    if args.domain is None:
        args.domain = sampler.domain
    if args.domain is None:
        print("ERROR: could not detect domain from checkpoint; pass --domain explicitly")
        sys.exit(1)
    elif args.domain != sampler.domain and sampler.domain is not None:
        print(f"WARNING: --domain={args.domain} but checkpoint domain={sampler.domain}")

    output_path = args.output
    if output_path is None:
        os.makedirs("artifacts", exist_ok=True)
        output_path = f"artifacts/validated_clusters_{args.domain}_k{args.num_clusters}.json"

    print(f"=== Stage 2: Cluster & Validate ===")
    print(f"  Checkpoint:       {args.checkpoint}")
    print(f"  Domain:           {args.domain}")
    print(f"  Model:            {sampler.model.n}-gram, vocab size {len(sampler.model.vocab)}")
    print(f"  Num clusters:     {args.num_clusters}")
    print(f"  Pool size:        {args.pool_size}")
    print(f"  Temperature:      {args.pool_temperature}")
    print(f"  Validation model: {args.validation_model}")
    print(f"  Max replacements: {args.max_replacements}")
    print(f"  Output:           {output_path}")
    print(f"  Started at:       {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    domain_config = DomainConfig(args.domain)
    length_dist, length_config = build_length_distribution(
        sampler,
        gaussian_mean=args.gaussian_mean,
        gaussian_std=args.gaussian_std,
        skew_alpha=args.skew_alpha,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    print(f"  Length distribution: mode={length_config.mode}, "
          f"range=[{args.min_length}, {args.max_length}]")

    action_types = ToolsSpecRetriever(path=domain_config.tool_spec_path).get_action_types()
    print(f"  Action types loaded: {len(action_types)} actions")

    if args.regular_edit_distance:
        sub_costs = SubstitutionCosts(
            cost_same_group=1.0,
            cost_same_type=1.0,
            cost_diff_type=1.0,
            group_map=None,
        )
        print(f"  Using regular (Levenshtein) edit distance — all substitution costs = 1.0")
    else:
        group_map = domain_config.action_group_map
        sub_costs = SubstitutionCosts(group_map=group_map)
        if group_map:
            print(f"  Using semantic group map ({len(group_map)} entries)")
        else:
            print(f"  Using default prefix-based grouping")

    # Domain-specific structural filter for pool generation.
    structural_filter = None
    if args.domain == "retail":
        from src.common.domain_validators.retail import retail_structural_filter
        structural_filter = retail_structural_filter
        print(f"  Using retail structural pre-filter for pool generation")
    elif args.domain == "telecom":
        from src.common.domain_validators.telecom import telecom_structural_filter
        structural_filter = telecom_structural_filter
        print(f"  Using telecom structural pre-filter for pool generation")

    if args.only_write:
        _base_filter = structural_filter
        def _write_only_filter(seq):
            if _base_filter is not None and not _base_filter(seq):
                return False
            return all(action_types.get(a) in ("WRITE", "GENERIC") for a in seq)
        structural_filter = _write_only_filter
        print(f"  Filtering to WRITE/GENERIC actions only (no READ)")

    print(f"\nGenerating pool of {args.pool_size} sequences...")
    clusterer = ActionSequenceClusterer(
        model=sampler.model,
        length_distribution=length_dist,
        length_config=length_config,
        n_clusters=args.num_clusters,
        pool_size=args.pool_size,
        temperature=args.pool_temperature,
        max_kmedoids_iters=args.max_kmedoids_iters,
        action_types=action_types,
        substitution_costs=sub_costs,
        structural_filter=structural_filter,
    )

    print(f"  Starting pool generation...")
    t_start = time.time()
    pool = clusterer.generate_pool()
    t_elapsed = time.time() - t_start
    print(f"  Generated {len(pool)} unique sequences in {t_elapsed:.1f}s")

    print(f"  Clustering {len(pool)} sequences into {args.num_clusters} clusters...")
    print(f"  Computing distance matrix ({len(pool)} x {len(pool)})...")
    t_dist_start = time.time()
    dist_matrix = clusterer.compute_distance_matrix(pool)
    t_dist = time.time() - t_dist_start
    print(f"    Distance matrix computed in {t_dist:.1f}s")

    print(f"  Running k-medoids algorithm...")
    t_kmed_start = time.time()
    medoids, labels = clusterer._run_kmedoids(dist_matrix, args.num_clusters, pool)
    t_kmed = time.time() - t_kmed_start
    print(f"    K-medoids completed in {t_kmed:.1f}s")
    seq_list = pool

    clusters = []
    for c_idx, med_idx in enumerate(medoids):
        member_indices = np.where(labels == c_idx)[0].tolist()
        member_dists = [(idx, float(dist_matrix[med_idx, idx])) for idx in member_indices]
        member_dists.sort(key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in member_dists]
        sorted_dists = [d for _, d in member_dists]
        sorted_seqs = [seq_list[idx] for idx in sorted_indices]
        clusters.append(ClusterInfo(
            cluster_idx=c_idx, medoid_idx=med_idx,
            medoid_sequence=seq_list[med_idx],
            member_indices=sorted_indices,
            member_sequences=sorted_seqs,
            intra_cluster_distances=sorted_dists,
        ))

    clusterer.sequences = seq_list
    clusterer.dist_matrix = dist_matrix
    clusterer.clusters = clusters

    stats = clusterer.summary_stats()
    print(f"  Pool: {stats['unique_sequences']} unique sequences from {stats['total_attempts']} attempts")
    print(f"  Cluster sizes: min={stats['cluster_size_min']}, "
          f"max={stats['cluster_size_max']}, "
          f"mean={stats['cluster_size_mean']:.1f}")

    validator = ActionSequenceValidator(
        domain=args.domain,
        model_name=f"vertex_ai/{args.validation_model}",
        max_output_tokens=args.max_output_tokens,
    )

    print(f"\nValidating {len(clusters)} medoids...")
    validation_results = validate_clusters(
        clusters,
        validator=validator,
        max_replacements=args.max_replacements,
    )

    status_counts = {"valid_medoid": 0, "replaced": 0, "unusable": 0}
    total_attempts = 0
    for vr in validation_results:
        status_counts[vr["status"]] = status_counts.get(vr["status"], 0) + 1
        total_attempts += vr["attempts"]

    print(f"\n--- Validation Summary ---")
    print(f"  Valid medoids:  {status_counts.get('valid_medoid', 0)}")
    print(f"  Replaced:       {status_counts.get('replaced', 0)}")
    print(f"  Unusable:       {status_counts.get('unusable', 0)}")
    print(f"  Total attempts: {total_attempts}")
    usable = status_counts.get("valid_medoid", 0) + status_counts.get("replaced", 0)
    print(f"  Usable clusters: {usable}/{len(clusters)}")

    # Re-cluster any clusters left unusable, up to max_recluster_rounds times.
    for round_num in range(args.max_recluster_rounds):
        unusable_cluster_indices = [
            vr["cluster_idx"] for vr in validation_results if vr["status"] == "unusable"
        ]
        if not unusable_cluster_indices:
            break

        print(f"\n--- Re-clustering round {round_num + 1}/{args.max_recluster_rounds} "
              f"({len(unusable_cluster_indices)} unusable) ---")
        print(f"  Unusable cluster indices: {unusable_cluster_indices}")

        frozen_seqs = [
            vr["final_sequence"]
            for vr in validation_results
            if vr["status"] != "unusable"
        ]
        n_new = len(unusable_cluster_indices)
        print(f"  Frozen: {len(frozen_seqs)} valid sequences, need {n_new} new clusters")

        print(f"  Generating new pool of {args.pool_size} sequences...")
        new_pool = clusterer.generate_pool()

        if len(new_pool) < n_new:
            print(f"  Not enough sequences ({len(new_pool)} < {n_new}), stopping.")
            break

        print(f"  Running constrained k-medoids...")
        new_cluster_infos = clusterer.recluster_with_frozen(frozen_seqs, n_new, new_pool)

        print(f"  Validating {len(new_cluster_infos)} new cluster candidates...")
        for ci, orig_idx in zip(new_cluster_infos, unusable_cluster_indices):
            ci.cluster_idx = orig_idx
        new_vrs = validate_clusters(
            new_cluster_infos,
            validator=validator,
            max_replacements=args.max_replacements,
        )
        for vr, orig_idx in zip(new_vrs, unusable_cluster_indices):
            vr["cluster_idx"] = orig_idx

        # Splice the new results/clusters back over the unusable slots.
        vr_by_idx = {vr["cluster_idx"]: vr for vr in validation_results}
        cluster_by_idx = {c.cluster_idx: c for c in clusters}
        for new_vr, new_ci in zip(new_vrs, new_cluster_infos):
            idx = new_vr["cluster_idx"]
            vr_by_idx[idx] = new_vr
            new_ci.cluster_idx = idx
            cluster_by_idx[idx] = new_ci

        validation_results = [vr_by_idx[i] for i in range(args.num_clusters)]
        clusters = [cluster_by_idx[i] for i in range(args.num_clusters)]

        status_counts = {"valid_medoid": 0, "replaced": 0, "unusable": 0}
        for vr in validation_results:
            status_counts[vr["status"]] = status_counts.get(vr["status"], 0) + 1
        still_unusable = status_counts.get("unusable", 0)
        usable_now = status_counts.get("valid_medoid", 0) + status_counts.get("replaced", 0)
        print(f"  After round {round_num + 1}: {usable_now}/{args.num_clusters} usable, "
              f"{still_unusable} unusable remaining")

    if args.max_recluster_rounds > 0:
        final_status_counts = {"valid_medoid": 0, "replaced": 0, "unusable": 0}
        for vr in validation_results:
            final_status_counts[vr["status"]] = final_status_counts.get(vr["status"], 0) + 1
        usable_final = (final_status_counts.get("valid_medoid", 0)
                        + final_status_counts.get("replaced", 0))
        print(f"\n--- Final Summary (after re-clustering) ---")
        print(f"  Valid medoids:  {final_status_counts.get('valid_medoid', 0)}")
        print(f"  Replaced:       {final_status_counts.get('replaced', 0)}")
        print(f"  Unusable:       {final_status_counts.get('unusable', 0)}")
        print(f"  Usable clusters: {usable_final}/{len(clusters)}")

    medoid_seqs = [vr["final_sequence"] for vr in validation_results if vr["final_sequence"]]
    if medoid_seqs:
        total_actions = sum(len(s) for s in medoid_seqs)
        total_writes = sum(
            1 for s in medoid_seqs for a in s if action_types.get(a) == "WRITE"
        )
        print(f"\n--- Write Action Summary ---")
        print(f"  Total actions across medoids: {total_actions}")
        print(f"  Total WRITE actions:          {total_writes}")
        print(f"  Overall write fraction:       {total_writes / total_actions:.2%}")

    clustering_config = {
        "pool_size": args.pool_size,
        "pool_temperature": args.pool_temperature,
        "max_kmedoids_iters": args.max_kmedoids_iters,
        "num_clusters": args.num_clusters,
        "max_recluster_rounds": args.max_recluster_rounds,
    }
    validation_config = {
        "validation_model": args.validation_model,
        "max_replacements": args.max_replacements,
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_validated_clusters(
        path=output_path,
        domain=args.domain,
        num_clusters=args.num_clusters,
        ngram_checkpoint=args.checkpoint,
        length_distribution_config=length_config.to_dict(),
        clustering_config=clustering_config,
        validation_config=validation_config,
        clusters=clusters,
        validation_results=validation_results,
    )

    t_total = time.time() - t_main_start
    print(f"\nArtifact saved to: {output_path}")
    print(f"Total elapsed time: {t_total:.1f}s ({t_total/60:.1f} minutes)")
    print(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
