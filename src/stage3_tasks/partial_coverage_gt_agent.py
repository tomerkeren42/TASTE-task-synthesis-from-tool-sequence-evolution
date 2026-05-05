"""GT agent variant that partially hides action arguments.

For a configurable fraction P of assistant actions (randomly selected),
argument values are redacted to `???`. The agent must derive those values
from the conversation rather than copying them from the action bank.

Coverage always hides at least 1 action: n_to_cover = max(1, floor(P * N_assistant)).
User actions are never covered — hiding what to instruct the user to do is meaningless.
Action presentation order is shuffled by default (shuffle=True) but can be kept in
original order (shuffle=False) to isolate the effect of argument coverage alone.

Prompt text is loaded from ``artifacts/prompts/stage3/partial_coverage_gt_agent.txt``.
"""
import math
import os
import random
import re
from typing import Dict, List, Optional

from tau2.agent.llm_agent import LLMGTAgent
from tau2.data_model.tasks import Action, Task
from tau2.environment.tool import Tool

from src.common.domain_utils import WORKSPACE_ROOT as _WORKSPACE_ROOT

_PROMPT_FILE = os.path.join(
    _WORKSPACE_ROOT, "artifacts", "prompts", "stage3", "partial_coverage_gt_agent.txt"
)

_SECTION_RE = re.compile(r"^===([A-Z_]+)===\s*$", re.MULTILINE)


def _load_prompt_sections(path: str) -> Dict[str, str]:
    """Parse a prompt file split into ``===SECTION_NAME===`` blocks."""
    with open(path, "r") as f:
        text = f.read()

    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        raise ValueError(f"No sections found in prompt file: {path}")

    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    required = {
        "SYSTEM_PROMPT",
        "AGENT_INSTRUCTION",
        "ORDERING_NOTE_SHUFFLED",
        "ORDERING_NOTE_FIXED",
        "WRITE_ONLY_NOTE",
        "ADVERSARIAL_USER_NOTE",
    }
    missing = required - sections.keys()
    if missing:
        raise ValueError(f"Prompt file {path} missing sections: {sorted(missing)}")
    return sections


_PROMPTS = _load_prompt_sections(_PROMPT_FILE)


class PartialCoverageGTAgent(LLMGTAgent):
    """GT agent that redacts argument values for a fraction of assistant actions.

    Args:
        coverage_p: Fraction of assistant actions to cover (redact). Default 0.33.
            At least 1 action is always covered: n_to_cover = max(1, floor(p * N_assistant)).
        shuffle: Whether to present actions in shuffled order. Default True.
            Set to False to isolate the effect of argument coverage from ordering difficulty.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        task: Task,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        provide_function_args: bool = True,
        coverage_p: float = 0.33,
        shuffle: bool = True,
        adversarial_user: bool = False,
        write_only: bool = False,
    ):
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            task=task,
            llm=llm,
            llm_args=llm_args,
            provide_function_args=provide_function_args,
        )
        self.coverage_p = coverage_p
        self.shuffle = shuffle
        self.adversarial_user = adversarial_user
        self.write_only = write_only

    @property
    def system_prompt(self) -> str:
        ordering_note = (
            _PROMPTS["ORDERING_NOTE_SHUFFLED"] if self.shuffle else _PROMPTS["ORDERING_NOTE_FIXED"]
        )
        instruction = _PROMPTS["AGENT_INSTRUCTION"].format(ordering_note=ordering_note)
        if self.write_only:
            instruction += "\n\n" + _PROMPTS["WRITE_ONLY_NOTE"]
        if self.adversarial_user:
            instruction += "\n\n" + _PROMPTS["ADVERSARIAL_USER_NOTE"]
        return _PROMPTS["SYSTEM_PROMPT"].format(
            agent_instruction=instruction,
            domain_policy=self.domain_policy,
            resolution_steps=self.make_agent_instructions_from_actions(),
        )

    def make_agent_instructions_from_actions(self) -> str:
        """Build action list (optionally shuffled) with argument values redacted for covered actions."""
        actions = list(self.task.evaluation_criteria.actions)

        # Only assistant actions are eligible for coverage
        assistant_indices = [i for i, a in enumerate(actions) if a.requestor == "assistant"]
        n_assistant = len(assistant_indices)
        # coverage_p == 0 means ZERO redaction (used for telecom validity runs).
        # For p > 0 we enforce at least one redaction so "partial coverage" is
        # meaningful; see module docstring.
        if n_assistant > 0 and self.coverage_p > 0.0:
            n_to_cover = max(1, math.floor(self.coverage_p * n_assistant))
            covered_indices = set(random.sample(assistant_indices, min(n_to_cover, n_assistant)))
        else:
            covered_indices = set()

        order = list(range(len(actions)))
        if self.shuffle:
            random.shuffle(order)

        lines = []
        for i in order:
            action = actions[i]
            if i in covered_indices:
                lines.append(f"- {self._make_covered_instruction(action)}")
            else:
                lines.append(
                    f"- {self.make_agent_instructions_from_action(action=action, include_function_args=self.provide_function_args)}"
                )
        return "\n".join(lines)

    @staticmethod
    def _make_covered_instruction(action: Action) -> str:
        """Render an assistant action with all argument values replaced by ???."""
        args_str = ", ".join(f"{k}=???" for k in action.arguments.keys())
        return f"Perform the following action: {action.name}({args_str})."
