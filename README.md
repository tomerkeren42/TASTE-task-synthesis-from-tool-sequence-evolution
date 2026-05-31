# A Matter of TASTE — Improving Coverage and Difficulty of Agent Benchmarks

> **Status: under review.**
> This repository accompanies a research paper that is currently under
> review. The code is shared in its current state for review and
> reproducibility purposes only. **Once the paper is published the
> repository will be open-sourced** under a permissive license; until then
> please treat this code as confidential and do not redistribute (see
> [License](#license)).

**TASTE turns a handful of seed tasks into a large, diverse, and difficult
tool-use benchmark.** Point it at a domain — its tools, its policy, and a few
example tasks — and it generates a fresh set of validated, runnable tasks
driven by the domain's *action space* rather than by hand-written scenarios.

Three reference domains ship ready to run: **`airline`**, **`retail`**, and
**`telecom`**.

### What you get

- **Scale** — grow a few seed tasks into hundreds of new ones.
- **Coverage** — tasks span the domain's action space, not just the patterns
  in your seeds.
- **Difficulty** — an optional adversarial stage plants traps that make tasks
  genuinely hard for agents.
- **Validated & executable** — every task is checked by a ground-truth agent
  running in the [tau2-bench](#place-tau2-bench) simulator, so it actually
  runs.

---

## How it works

The pipeline has three stages. Each consumes the output of the previous one.

```
  seed tasks (per-domain tasks.json)
        │
        ▼
  ┌─ Stage 1 ─ n-gram model ──────────────────────────────────────────┐
  │  Learns which action sequences are plausible in the domain from    │
  │  the seeds (a Bayesian signed n-gram model), then samples and      │
  │  LLM-validates new sequences to refine it.                         │
  │  → artifacts/ngram/checkpoints/ngram_checkpoint_<domain>_*.json    │
  └────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Stage 2 ─ cluster ───────────────────────────────────────────────┐
  │  Samples a large pool of new sequences, clusters them with         │
  │  k-medoids (type/group-weighted edit distance) for diversity, and  │
  │  LLM-validates one representative (medoid) per cluster.            │
  │  → artifacts/validated_clusters_<domain>_k<K>.json                 │
  └────────────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌─ Stage 3 ─ tasks ─────────────────────────────────────────────────┐
  │  generate: authors a full runnable task per cluster (description,  │
  │            scenario, DB state, eval criteria) and validates it.    │
  │  evolve  : (optional) adversarially rewrites tasks to be harder.   │
  │  → task_sets/<name>/tasks.json                                     │
  └────────────────────────────────────────────────────────────────────┘
```

---

## Setup

### Requirements

- Python **≥ 3.10**
- A `tau2-bench` checkout (see below) — no install needed
- Access to an LLM backend (Vertex AI by default; any
  [`litellm`](https://github.com/BerriAI/litellm)-compatible model works)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .          # installs the package and all runtime deps
```

All runtime dependencies are declared in `pyproject.toml`, so `pip install -e .`
is all you need.

### Place `tau2-bench`

The pipeline reuses tau2-bench's domain definitions, environment, and
`run_task` simulator for ground-truth validation. Check it out at
`./tau2-bench/` — the entry points add `tau2-bench/src` to `sys.path`
automatically, so no separate install is required.

```bash
git clone <tau2-bench-repo-url> tau2-bench
```

### Authenticate the LLM provider

For Vertex AI (the default):

```bash
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=<your-project>
```

For other providers, set the environment variables expected by `litellm` and
pass the corresponding model with `--validation-model` / `--model-name` /
`--gt-llm`.

---

## Quickstart — end-to-end on the airline domain

Run everything from the repository root via `python -m`. The four commands
below take the `airline` seeds all the way to an adversarially hard task set.

```bash
# Stage 1 — train the n-gram model
python -m src.stage1_ngram.initialize_ngram_model \
    --domain airline \
    --num-samples 3000

# Stage 2 — cluster + LLM-validate medoids
python -m src.stage2_cluster.cluster_and_validate \
    --checkpoint artifacts/ngram/checkpoints/ngram_checkpoint_airline_3000samples.json \
    --domain airline \
    -k 20 --pool-size 2000 \
    --output artifacts/validated_clusters_airline_k20.json

# Stage 3a — generate runnable tasks from the validated clusters
python -m src.stage3_tasks.generate_tasks \
    --clusters artifacts/validated_clusters_airline_k20.json \
    --domain airline \
    --name airline_v1
# → task_sets/airline_v1/

# Stage 3b (optional) — adversarially evolve the generated task set
python -m src.stage3_tasks.evolve_tasks \
    --task-set airline_v1 \
    --output-name airline_adversarial_v1 \
    --domain airline
# → task_sets/airline_adversarial_v1/
```

The generated benchmark lives in `task_sets/<name>/tasks.json`. Pass `--help`
to any entry point for the full list of flags.

> **Tip:** Stages 3a and 3b checkpoint after every task. Re-run with
> `--resume <name>` to pick up where an interrupted run left off.

---

## Stage reference

Each stage is a single entry point. Only the most useful flags are listed —
run with `--help` for the complete set.

### Stage 1 — `src.stage1_ngram.initialize_ngram_model`

Trains a Bayesian signed n-gram model over the seed action sequences, writing a
final checkpoint under `artifacts/ngram/checkpoints/` for stage 2.

| Flag | Default | Purpose |
|------|---------|---------|
| `--domain` | `airline` | Which domain to train on |
| `--num-samples` | `3000` | Total sequences to sample/validate |
| `--seed-tasks` | domain default | Use a custom `tasks.json` as seeds |
| `--validation-model` | `gemini/gemini-3-flash-preview` | LLM used to validate sampled sequences |
| `--n` | `3` | N-gram order |
| `--no-ingest-negatives` | off | Don't learn from invalid sequences |

Model-tuning knobs: `--alpha0`, `--lambda-neg`, `--tau0`,
`--tau-decay-steps`, `--negative-weight`, `--save-every`, `--save-dir`.

### Stage 2 — `src.stage2_cluster.cluster_and_validate`

Generates a pool of new sequences, clusters them under a type/group-weighted
edit distance, LLM-validates each medoid, and re-clusters any unusable clusters.

| Flag | Default | Purpose |
|------|---------|---------|
| `--checkpoint` | **required** | Stage-1 checkpoint to sample from |
| `-k` / `--num-clusters` | **required** | Number of clusters (≈ number of tasks) |
| `--domain` | auto-detected | Override domain auto-detection |
| `--pool-size` | `2000` | How many sequences to sample before clustering |
| `--output` | auto | Where to write the validated clusters |

Length distribution (skew-normal): `--gaussian-mean`, `--gaussian-std`,
`--skew-alpha`, `--min-length`, `--max-length`. Other knobs:
`--pool-temperature`, `--max-replacements`, `--max-recluster-rounds`,
`--validation-model`, `--only-write`, `--regular-edit-distance`.

### Stage 3a — `src.stage3_tasks.generate_tasks`

Asks an LLM to author a complete task per medoid, then validates it with a
partial-coverage ground-truth agent. Failures are patched, regenerated, and
finally re-clustered.

| Flag | Default | Purpose |
|------|---------|---------|
| `--clusters` | **required** | Stage-2 output to build tasks from |
| `--domain` | `airline` | Domain name |
| `--name` | auto | Output task-set name (`task_sets/<name>/`) |
| `--model-name` | `gemini/gemini-3-flash-preview` | LLM that authors tasks |
| `--gt-llm` | `gemini/gemini-3-flash-preview` | LLM driving the GT validation agent |
| `--resume` | — | Resume a previous run (skips finished clusters) |

Other knobs: `--gt-coverage-p`, `--gt-coverage-shuffle`,
`--max-task-retries`, `--max-patch-attempts`, `--indices`, `--write-only`
(telecom).

### Stage 3b — `src.stage3_tasks.evolve_tasks`

Three-phase adversarial evolution over an existing task set:

1. **Strategy** — analyze the golden actions + DB to plan an adversarial twist.
2. **DB traps** — synthesize plausible-looking decoy entities that mislead the
   agent.
3. **Scenario** — rewrite the user instructions to encode the trap.

Each evolved task is re-validated by the GT agent; on failure it falls back to a
"lite" version, then to the original.

| Flag | Default | Purpose |
|------|---------|---------|
| `--task-set` | **required** | Task set to evolve (path or name under `task_sets/`) |
| `--output-name` | `<input>_adversarial` | Output directory name |
| `--domain` | `airline` | Domain name |
| `--model-name` | `vertex_ai/gemini-3-flash-preview` | LLM for all 3 phases |
| `--gt-llm` | `vertex_ai/gemini-3-flash-preview` | LLM driving GT validation |
| `--resume` | off | Resume an interrupted run |

Other knobs: `--max-phase1-retries`, `--max-phase3-retries`, `--indices`.

---

## Adding your own domain

A domain is fully described by four files under `artifacts/domains/<name>/`:

| File | Purpose |
|------|---------|
| `domain.json` | Wiring: `data_model_module`, `db_class`, `environment_module`, optional `action_group_map`, optional `user_data_model_module` / `user_db_class` |
| `policy.md` | Natural-language policy shown to the agent and used as LLM context |
| `tool_spec.json` | Tool schema with action types (`READ` / `WRITE` / `GENERIC`) |
| `tasks.json` | A small set of seed tasks that bootstrap the n-gram model |

You also need a matching `tau2-bench` domain exposing the modules referenced in
`domain.json`. Optionally add per-domain prompt overrides under
`artifacts/prompts/stage{2,3}/<name>/`.

---

## Outputs at a glance

| Stage | Default location |
|-------|------------------|
| 1 | `artifacts/ngram/training/<domain>/...`<br>`artifacts/ngram/checkpoints/ngram_checkpoint_<domain>_<N>samples.json` |
| 2 | `artifacts/validated_clusters_<domain>_k<K>.json` (override with `--output`) |
| 3a | `task_sets/<name>/{tasks.json, generation_meta.json, progress.jsonl}` |
| 3b | `task_sets/<name>/{tasks.json, metadata.json, strategies.jsonl}` |

---

## Repository layout

```
.
├── src/
│   ├── stage1_ngram/initialize_ngram_model.py   # entry point: stage 1
│   ├── stage2_cluster/cluster_and_validate.py   # entry point: stage 2
│   ├── stage3_tasks/
│   │   ├── generate_tasks.py                     # entry point: stage 3a
│   │   ├── evolve_tasks.py                        # entry point: stage 3b
│   │   └── ...                                    # task building, evolution
│   └── common/                                    # LLM calls, prompts,
│       ├── domain_validators/                     # domain config, sampler
│       └── sampler/                               # n-gram model + clustering
│
├── artifacts/
│   ├── domains/<domain>/    # policy.md, tool_spec.json, domain.json, tasks.json
│   ├── prompts/             # stage2 + stage3 prompts (optional per-domain)
│   ├── ngram/               # stage-1 outputs
│   ├── validated_clusters/  # stage-2 outputs
│   └── task_sets/           # historical stage-3 outputs
│
├── task_sets/               # default output dir for new runs
└── tau2-bench/              # external dependency (see Setup)
```

---

## Citation

A BibTeX entry will be added here once the paper is published. Until then,
please contact the authors before citing or building on this work.

## License

See [`LICENSE`](LICENSE). This repository is shared **for review and
reproduction purposes only** while the paper is under review; please do not
redistribute. A permissive open-source license will replace it at the time of
public release.
