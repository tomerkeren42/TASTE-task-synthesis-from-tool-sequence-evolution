# A Matter of TASTE — Improving Coverage and Difficulty of Agent Benchmarks

> **Status: pre-publication.**
> This repository accompanies a research paper that has not yet been
> published. The code is shared in its current state for review and
> reproducibility purposes only. **Once the paper is published the
> repository will be open-sourced** under a permissive license; until then
> please treat this code as confidential and do not redistribute.

`a-matter-of-taste` characterizes tool-use agent benchmarks and provides a
three-stage pipeline that turns a small set of seed tasks into a much larger,
more diverse, and more difficult benchmark — all driven from the action-space
of the underlying domain rather than from hand-written scenarios.

---

## How the pipeline works

The repository ships four entry points, organized into three stages. Each
stage consumes the output of the previous one.

```
seed tasks
   │   (per-domain tasks.json)
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1 — Initialize n-gram model                                   │
│   src.stage1_ngram.initialize_ngram_model                           │
│   Trains a Bayesian signed n-gram model over the seed action        │
│   sequences; samples + LLM-validates new sequences and ingests      │
│   them as positives / negatives.                                    │
│                                                                     │
│   → artifacts/ngram/checkpoints/ngram_checkpoint_<domain>_*.json    │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 2 — Cluster generated sequences and validate medoids          │
│   src.stage2_cluster.cluster_and_validate                           │
│   Generates a large pool of new sequences, clusters them with       │
│   k-medoids under a type/group-weighted edit distance, then         │
│   LLM-validates each cluster's medoid (with replacement search +    │
│   re-clustering for unusable clusters).                             │
│                                                                     │
│   → artifacts/validated_clusters_<domain>_k<K>.json                 │
└─────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 3 — Generate, validate, and (optionally) evolve tasks         │
│   src.stage3_tasks.generate_tasks                                   │
│     LLM-authors a full task (description, user_scenario, DB         │
│     init, evaluation criteria) per validated cluster, then          │
│     validates with a partial-coverage GT agent. Patches and         │
│     re-clusters on failure.                                         │
│                                                                     │
│   src.stage3_tasks.evolve_tasks                                     │
│     Three-phase adversarial evolution (strategy → DB traps →        │
│     scenario rewrite) over an existing task set.                    │
│                                                                     │
│   → task_sets/<name>/tasks.json                                     │
│   → task_sets/<name>/generation_meta.json | metadata.json           │
└─────────────────────────────────────────────────────────────────────┘
```

Three reference domains are included out of the box: **`airline`**,
**`retail`**, and **`telecom`** (see `artifacts/domains/<domain>/`).

---

## Repository layout

```
.
├── src/
│   ├── stage1_ngram/
│   │   └── initialize_ngram_model.py   # entry point: stage 1
│   ├── stage2_cluster/
│   │   ├── cluster_and_validate.py     # entry point: stage 2
│   │   ├── action_sequence_validator.py
│   │   └── medoid_validation.py
│   ├── stage3_tasks/
│   │   ├── generate_tasks.py           # entry point: stage 3 (generate)
│   │   ├── evolve_tasks.py             # entry point: stage 3 (evolve)
│   │   ├── task_validator.py
│   │   ├── task_generation.py
│   │   ├── task_builder.py
│   │   ├── adversarial_evolver.py
│   │   ├── env_assertion_synthesizer.py
│   │   ├── scenario_aligner.py
│   │   └── ...
│   └── common/                         # shared across stages
│       ├── call_to_llm.py              # LLMCaller (Vertex AI / litellm)
│       ├── prompt_manager.py
│       ├── llm_response_parser.py
│       ├── domain_config.py
│       ├── domain_utils.py             # WORKSPACE_ROOT lives here
│       ├── tool_spec_retriever.py
│       ├── flight_ambiguity_checker.py
│       ├── domain_validators/          # airline, retail, telecom
│       └── sampler/                    # n-gram model + clustering
│
├── artifacts/
│   ├── domains/<domain>/               # policy.md, tool_spec.json,
│   │                                   # domain.json, seed tasks.json
│   ├── prompts/
│   │   ├── stage2/                     # one shared prompt across domains
│   │   └── stage3/                     # task-generation + evolution
│   │       └── <domain>/               # optional per-domain overrides
│   ├── ngram/                          # stage-1 outputs
│   ├── validated_clusters/             # stage-2 outputs
│   └── task_sets/                      # historical stage-3 outputs
│
├── task_sets/                          # default output dir for new runs
└── tau2-bench/                         # external dependency (see Setup)
```

---

## Setup

### Requirements

- Python **≥ 3.10**
- `tau2-bench` checked out at `./tau2-bench/` (the entry points add
  `tau2-bench/src` to `sys.path` automatically — no install required)
- Access to one of the supported LLM backends (Vertex AI by default; any
  `litellm`-compatible model works)

### Install

```bash
# 1. Create and activate a virtual env
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the project (editable so local edits show up immediately)
pip install -e .

# 3. Install the project's runtime dependencies
pip install loguru numpy scipy litellm vertexai pydantic
```

Some optional features (e.g. specific LLM providers) may pull in additional
extras — install them on demand based on the `--validation-model` /
`--gt-llm` you choose.

### Authenticate the LLM provider

For Vertex AI (the default):

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-project>
```

For other providers, set the environment variables expected by
[`litellm`](https://github.com/BerriAI/litellm).

### Place `tau2-bench`

```bash
git clone <tau2-bench-repo-url> tau2-bench
```

The pipeline reuses tau2-bench's domain definitions, environment, and the
`run_task` simulator for GT-agent validation.

---

## Quickstart — end-to-end on the airline domain

All entry points are run from the repository root via `python -m`.

```bash
# Stage 1 — train the n-gram model (writes a final-state checkpoint)
python -m src.stage1_ngram.initialize_ngram_model \
    --domain airline \
    --num-samples 3000

# Stage 2 — cluster + LLM-validate medoids
python -m src.stage2_cluster.cluster_and_validate \
    --checkpoint artifacts/ngram/checkpoints/ngram_checkpoint_airline_3000samples.json \
    --domain airline \
    -k 20 --pool-size 2000 \
    --output artifacts/validated_clusters_airline_k20.json

# Stage 3a — generate tasks from the validated clusters
python -m src.stage3_tasks.generate_tasks \
    --clusters artifacts/validated_clusters_airline_k20.json \
    --domain airline \
    --name airline_easy_v1
# → task_sets/airline_easy_v1/{tasks.json, generation_meta.json, progress.jsonl}

# Stage 3b (optional) — adversarially evolve a generated task set
python -m src.stage3_tasks.evolve_tasks \
    --task-set airline_easy_v1 \
    --output-name airline_adversarial_v1 \
    --domain airline
# → task_sets/airline_adversarial_v1/{tasks.json, metadata.json, strategies.jsonl}
```

Pass `--help` to any entry point for the full list of flags.

---

## Stage-by-stage reference

### Stage 1 — `src.stage1_ngram.initialize_ngram_model`

Trains a Bayesian signed n-gram model with periodic checkpoints
(`pre_seed`, `post_seed`, periodic, `final`) and writes a descriptive
final-state copy under `artifacts/ngram/checkpoints/` for stage 2 to consume.

Key flags: `--domain`, `--num-samples`, `--save-every`, `--n`,
`--alpha0`, `--lambda-neg`, `--tau0`, `--tau-decay-steps`,
`--negative-weight`, `--no-ingest-negatives`,
`--validation-model`, `--seed-tasks`.

### Stage 2 — `src.stage2_cluster.cluster_and_validate`

Loads a stage-1 checkpoint, generates a pool of new sequences, runs
k-medoids under a type/group-weighted edit distance, LLM-validates each
medoid, and re-clusters any unusable clusters up to
`--max-recluster-rounds` times.

Length sampling is always **skew-normal** — tune via
`--gaussian-mean`, `--gaussian-std`, `--skew-alpha`, `--min-length`,
`--max-length`. Other flags: `--pool-size`, `--pool-temperature`,
`-k/--num-clusters`, `--max-replacements`, `--validation-model`,
`--only-write` (telecom), `--regular-edit-distance`.

### Stage 3a — `src.stage3_tasks.generate_tasks`

For each usable medoid, asks an LLM to author a complete task
(`description`, `user_scenario`, initial DB state, evaluation criteria),
then validates via a partial-coverage GT agent. Failures are first patched
in place, then fully regenerated, and finally re-clustered.

Successful tasks are checkpointed into `task_sets/<name>/tasks.json` after
every success — re-run with `--resume <name>` (or `--resume`) to pick up
where a previous run left off, skipping any `cluster_idx` already on disk.

Key flags: `--clusters` (required), `--domain`, `--model-name`,
`--gt-llm`, `--gt-coverage-p`, `--gt-coverage-shuffle`,
`--max-task-retries`, `--max-patch-attempts`, `--name`, `--resume`,
`--indices`, `--write-only` (telecom).

### Stage 3b — `src.stage3_tasks.evolve_tasks`

Three-phase adversarial evolution over an existing task set:

1. **Strategy** — analyze golden actions + DB to plan an adversarial twist.
2. **DB traps** — synthesize decoy entities that look plausible but lead the
   agent astray.
3. **Scenario** — rewrite the user instructions to encode the trap.

Each evolved task is re-validated via the GT agent; failures fall back to a
"lite" version, and finally to the original. Output goes to
`task_sets/<output-name>/`.

Key flags: `--task-set` (required), `--output-name`, `--domain`,
`--model-name`, `--gt-llm`, `--max-phase1-retries`, `--max-phase3-retries`,
`--resume`, `--indices`.

---

## Adding a new domain

A domain is fully described by four files under `artifacts/domains/<name>/`:

| File             | Purpose |
|------------------|---------|
| `domain.json`    | Wiring: `data_model_module`, `db_class`, `environment_module`, optional `action_group_map`, optional `user_data_model_module`/`user_db_class` |
| `policy.md`      | Natural-language policy shown to the agent and used as LLM context |
| `tool_spec.json` | Tool schema with action types (`READ`/`WRITE`/`GENERIC`) |
| `tasks.json`     | A small set of seed tasks — these bootstrap the n-gram model |

You will also need a `tau2-bench` domain that exposes the modules referenced
from `domain.json`. Optionally drop per-domain prompt overrides into
`artifacts/prompts/stage{2,3}/<name>/`.

---

## Outputs at a glance

| Stage | Default location |
|-------|------------------|
| 1     | `artifacts/ngram/training/<domain>/{pre_seed,post_seed,*,*_final}.json`<br>`artifacts/ngram/checkpoints/ngram_checkpoint_<domain>_<N>samples.json` |
| 2     | `artifacts/validated_clusters_<domain>_k<K>.json` (override with `--output`) |
| 3a    | `task_sets/<name>/{tasks.json, generation_meta.json, progress.jsonl}` |
| 3b    | `task_sets/<name>/{tasks.json, metadata.json, strategies.jsonl}` |

---

## Citation

A BibTeX entry will be added here once the paper is published. Until then,
please contact the authors before citing or building on this work.

---

## License

License will be added at the time of public release alongside the paper.
Until then this repository is shared **for review and reproduction
purposes only**; please do not redistribute.
