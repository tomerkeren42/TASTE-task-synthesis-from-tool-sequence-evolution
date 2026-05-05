"""
Prompt Manager

Loads and manages prompt templates for task generation.

Layout (under ``artifacts/prompts/``):
    stage2/                         # cluster validation (stage 2)
        validate_action_sequence.txt   # single, domain-agnostic prompt
    stage3/                         # task generation + adversarial evolution
        create_user_task.txt
        generate_db_initialization.txt
        task_coherence_review.txt
        patch_task.txt
        patch_db.txt
        generate_env_assertions.txt
        align_scenario_with_end_state.txt
        partial_coverage_gt_agent.txt
        db_trap_construction[_<domain>].txt
        adversarial_strategy[_<domain>].txt
        adversarial_scenario[_<domain>].txt
        adversarial_scenario_lite[_<domain>].txt
        <domain>/                   # per-domain overrides
            create_user_task.txt
            ...

Stage-3 defaults live at the stage root and are overridden by any file with
the same basename inside the matching ``<domain>/`` subdir. Stage 2 uses a
single shared prompt for all domains -- the policy and tool spec passed at
call time supply the domain-specific context.
"""

import os
from typing import Dict, Optional

# Stage 2: shared, domain-agnostic prompts (no per-domain override pattern).
_STAGE2_PROMPT_FILES = {
    'validate_action_sequence': 'validate_action_sequence.txt',
}

# Stage 3: prompts that have a per-domain override pattern.
_STAGE3_PROMPT_FILES = {
    'create_user_task': 'create_user_task.txt',
    'generate_db_initialization': 'generate_db_initialization.txt',
    'task_coherence_review': 'task_coherence_review.txt',
    'patch_task': 'patch_task.txt',
    'patch_db': 'patch_db.txt',
    'generate_env_assertions': 'generate_env_assertions.txt',
    'align_scenario_with_end_state': 'align_scenario_with_end_state.txt',
}

# Stage 3: shared, domain-agnostic prompts for adversarial evolution.
# These ship as flat files; the policy + tool spec + DB schema/summary supply
# the per-domain context at call time, just like ``validate_action_sequence``.
_STAGE3_FLAT_PROMPT_FILES = {
    'db_trap_construction': 'db_trap_construction.txt',
    'adversarial_strategy': 'adversarial_strategy.txt',
    'adversarial_scenario': 'adversarial_scenario.txt',
    'adversarial_scenario_lite': 'adversarial_scenario_lite.txt',
}


class PromptManager:
    """Loads and manages prompt templates."""

    def __init__(self, prompts_dir: str, domain: Optional[str] = None):
        """
        Args:
            prompts_dir: Root prompts directory (e.g. ``artifacts/prompts/``).
                         Must contain ``stage2/`` and ``stage3/`` subdirs.
            domain: Domain name (e.g. ``"retail"``). If provided, prompts
                    inside ``stage*/<domain>/`` override the stage defaults.
        """
        self.prompts_dir = prompts_dir
        self.domain = domain
        self.prompts: Dict[str, str] = {}
        self._load_prompts()

    def _load_flat(self, stage_dir: str, files: Dict[str, str]) -> None:
        for prompt_name, filename in files.items():
            path = os.path.join(stage_dir, filename)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.prompts[prompt_name] = f.read()

    def _load_with_domain_override(self, stage_dir: str, files: Dict[str, str]) -> None:
        for prompt_name, filename in files.items():
            base_path = os.path.join(stage_dir, filename)
            if os.path.exists(base_path):
                with open(base_path, 'r') as f:
                    self.prompts[prompt_name] = f.read()

            if self.domain is not None:
                override_path = os.path.join(stage_dir, self.domain, filename)
                if os.path.exists(override_path):
                    with open(override_path, 'r') as f:
                        self.prompts[prompt_name] = f.read()

    def _load_prompts(self) -> None:
        stage2_dir = os.path.join(self.prompts_dir, 'stage2')
        stage3_dir = os.path.join(self.prompts_dir, 'stage3')

        # Stage 2 prompts are shared across all domains -- no per-domain override.
        self._load_flat(stage2_dir, _STAGE2_PROMPT_FILES)

        # Stage 3 prompts: defaults at the root, optional per-domain overrides.
        self._load_with_domain_override(stage3_dir, _STAGE3_PROMPT_FILES)

        # Stage 3 evolve/adversarial prompts ship as flat files keyed by suffix.
        self._load_flat(stage3_dir, _STAGE3_FLAT_PROMPT_FILES)

    def get_prompt(self, prompt_name: str) -> str:
        """
        Get a prompt by name.

        Raises:
            KeyError: If prompt name not found
        """
        if prompt_name not in self.prompts:
            raise KeyError(
                f"Prompt '{prompt_name}' not found. "
                f"Available prompts: {list(self.prompts.keys())}"
            )
        return self.prompts[prompt_name]
