"""Align the user_scenario of a telecom task with the gold-replay end state.

Tasks fail when the scenario tells the user-LLM to pursue an outcome that the
golden action sequence can't actually achieve (e.g., scenario says "my data
isn't working, restore it" but the goldens leave the device in
`service='no_service'`). This module uses the gold-env end state — already
computed by the EnvAssertionSynthesizer — to rewrite reason_for_call /
known_info / task_instructions so the user-LLM will naturally drive the
conversation toward the end state the goldens produce.

Contract:
- Never-raises. On any LLM/parse/verification failure, leaves the scenario
  untouched and returns False. The caller proceeds to GT validation with the
  original scenario; a brittle alignment step must not fail an otherwise-good
  task. Failures are logged under logs/telecom/scenario_alignment_errors/.
- Telecom-only. Constructed once per generation run and called per task.
- Mutates the passed task_dict in place when alignment succeeds.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from src.common.domain_config import DomainConfig
from src.common.domain_utils import WORKSPACE_ROOT
from src.stage3_tasks.env_assertion_synthesizer import EnvAssertionSynthesizer
from src.common.llm_response_parser import LLMResponseParser
from src.common.prompt_manager import PromptManager


_LOGS_BASE = os.path.join(WORKSPACE_ROOT, "logs")

_REQUIRED_PHRASES = (
    "Wait for the agent to confirm completion of each action",
    "Whenever the agent asks you about your device",
)


class ScenarioAligner:
    """Refines user_scenario to match the gold-replay end state.

    Requires an LLMCaller. Loads its prompt template from the shared
    ``artifacts/prompts/stage3/`` directory (with a per-domain override at
    ``artifacts/prompts/stage3/telecom/`` when present) and uses the existing
    EnvAssertionSynthesizer to compute end-state snapshots.
    """

    def __init__(
        self,
        domain_config: DomainConfig,
        llm_caller: Any,
        prompt_name: str = "align_scenario_with_end_state",
    ):
        if domain_config.domain != "telecom":
            raise ValueError(
                f"ScenarioAligner is telecom-only; got domain={domain_config.domain}"
            )
        self._domain_config = domain_config
        self._llm = llm_caller
        self._synth = EnvAssertionSynthesizer(domain_config)
        self._pm = PromptManager(prompts_dir=domain_config.prompts_dir, domain=domain_config.domain)
        self._prompt_name = prompt_name
        self._parser = LLMResponseParser()

    # -- public -------------------------------------------------------

    def align(self, task_dict: Dict[str, Any]) -> bool:
        """Rewrite ``task_dict['user_scenario']['instructions']`` if the
        scenario doesn't align with the gold-replay end state.

        Returns True if the scenario was rewritten. False if alignment was
        skipped or failed (scenario untouched in both cases).
        """
        try:
            end_state = self._compute_end_state(task_dict)
        except Exception as e:
            self._log_failure(task_dict, "replay_failed", str(e))
            return False

        try:
            prompt = self._build_prompt(task_dict, end_state)
        except Exception as e:
            self._log_failure(task_dict, "prompt_build_failed", str(e))
            return False

        try:
            response = self._llm.call(prompt, stage_name="align_scenario")
        except Exception as e:
            self._log_failure(task_dict, "llm_failed", str(e))
            return False

        try:
            obj = self._parser.extract_json(response)
        except Exception as e:
            self._log_failure(task_dict, "parse_failed", str(e), response=response)
            return False

        if not self._looks_valid(obj):
            self._log_failure(task_dict, "invalid_shape", repr(obj)[:500], response=response)
            return False

        # Apply mutation
        inst = task_dict.setdefault("user_scenario", {}).setdefault("instructions", {})
        inst["reason_for_call"] = obj["reason_for_call"]
        inst["known_info"] = obj["known_info"]
        inst["task_instructions"] = obj["task_instructions"]
        if obj.get("description_purpose") and task_dict.get("description"):
            task_dict["description"]["purpose"] = obj["description_purpose"]
        return True

    # -- internals ---------------------------------------------------

    def _compute_end_state(self, task_dict: Dict[str, Any]) -> Dict[str, Any]:
        env, err = self._synth._build_env_and_replay(task_dict)
        if err is not None:
            raise RuntimeError(f"gold replay failed: {err}")
        return self._synth._capture_end_state(env)

    def _build_prompt(
        self, task_dict: Dict[str, Any], end_state: Dict[str, Any]
    ) -> str:
        template = self._pm.get_prompt(self._prompt_name)

        init = task_dict.get("initial_state") or {}
        init_data = init.get("initialization_data") or {}
        agent = init_data.get("agent_data") or {}
        user = init_data.get("user_data") or {}
        customers = agent.get("customers") or []
        lines = agent.get("lines") or []
        bills = agent.get("bills") or []

        c0 = customers[0] if customers else {}
        l0 = lines[0] if lines else {}
        dev_start = (user.get("device") or {})
        sur_start = (user.get("surroundings") or {})
        apn_start = (dev_start.get("active_apn_settings") or {}).get("apn_name")
        sig_start = sur_start.get("signal_strength") or {}

        ga = task_dict.get("evaluation_criteria", {}).get("actions", []) or []
        assertions = task_dict.get("evaluation_criteria", {}).get("env_assertions") or []

        ast = end_state.get("assistant_state") or {}
        dev_end = (end_state.get("user_state") or {}).get("device") or {}
        derived = end_state.get("derived") or {}
        apn_end = (dev_end.get("active_apn_settings") or {}).get("apn_name")

        # Pull the target line's final status from the replayed assistant state.
        # Fallback: first line in the task init (this matches how lookups are
        # scoped elsewhere — a task always operates on the customer's lines).
        target_line_id = l0.get("line_id")
        for a in ga:
            if a.get("arguments", {}).get("line_id"):
                target_line_id = a["arguments"]["line_id"]
                break
        line_status_end = None
        if target_line_id and ast.get("lines"):
            line_status_end = ast["lines"].get(target_line_id, {}).get("status")

        current_inst = (
            (task_dict.get("user_scenario") or {}).get("instructions") or {}
        )

        return template.format(
            identity_name=c0.get("full_name", ""),
            identity_phone=c0.get("phone_number", ""),
            identity_customer_id=c0.get("customer_id", ""),
            identity_line_id=l0.get("line_id", ""),
            identity_bill_ids=", ".join(b.get("bill_id", "") for b in bills) or "(none)",
            line_status_start=l0.get("status"),
            service_start=dev_start.get("network_connection_status"),
            airplane_start=dev_start.get("airplane_mode"),
            data_enabled_start=dev_start.get("data_enabled"),
            roaming_start=dev_start.get("roaming_enabled"),
            apn_start=apn_start,
            is_abroad=sur_start.get("is_abroad"),
            roaming_allowed=sur_start.get("roaming_allowed"),
            data_exceeded=sur_start.get("mobile_data_usage_exceeded"),
            signal_4g=sig_start.get("4G"),
            signal_5g=sig_start.get("5G"),
            golden_actions_block=_format_actions(ga),
            line_status_end=line_status_end,
            service_end=dev_end.get("network_connection_status"),
            airplane_end=dev_end.get("airplane_mode"),
            data_enabled_end=dev_end.get("data_enabled"),
            roaming_end=dev_end.get("roaming_enabled"),
            apn_end=apn_end,
            mobile_data_working_end=derived.get("mobile_data_working"),
            can_send_mms_end=derived.get("can_send_mms"),
            speed_end=derived.get("speed_test"),
            assertions_block=_format_assertions(assertions),
            current_reason=current_inst.get("reason_for_call", ""),
            current_known_info=current_inst.get("known_info", ""),
            current_instructions=current_inst.get("task_instructions", ""),
        )

    @staticmethod
    def _looks_valid(obj: Any) -> bool:
        if not isinstance(obj, dict):
            return False
        for k in ("reason_for_call", "known_info", "task_instructions"):
            if not isinstance(obj.get(k), str) or not obj[k].strip():
                return False
        ti = obj["task_instructions"]
        return all(p in ti for p in _REQUIRED_PHRASES)

    def _log_failure(
        self,
        task_dict: Dict[str, Any],
        kind: str,
        detail: str,
        response: Optional[str] = None,
    ) -> None:
        log_dir = os.path.join(_LOGS_BASE, "telecom", "scenario_alignment_errors")
        os.makedirs(log_dir, exist_ok=True)
        task_id = task_dict.get("id", "unknown")
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(task_id))
        path = os.path.join(log_dir, f"{safe}_{int(time.time())}_{kind}.json")
        with open(path, "w") as fh:
            json.dump(
                {"task_id": task_id, "kind": kind, "detail": detail, "response": response},
                fh,
                indent=2,
                default=str,
            )


def _format_actions(actions: List[Dict[str, Any]]) -> str:
    if not actions:
        return "  (none)"
    lines = []
    for i, a in enumerate(actions):
        lines.append(
            f'  {i}. [{a.get("requestor","?")}] {a.get("name","?")} {a.get("arguments",{})}'
        )
    return "\n".join(lines)


def _format_assertions(assertions: List[Dict[str, Any]]) -> str:
    if not assertions:
        return "  (none)"
    lines = []
    for a in assertions:
        lines.append(
            f'  [{a.get("env_type","?")}] {a.get("func_name","?")} {a.get("arguments",{})}'
        )
    return "\n".join(lines)
