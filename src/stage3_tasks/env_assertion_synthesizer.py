"""Rule-based synthesis of env_assertions for telecom tasks.

The synthesizer replays the task's golden actions on a fresh environment,
captures the resulting end state, and then selects env_assertions
programmatically — no LLM call, no value-synthesis errors. Values are read
directly from the observed env state, so verification after construction is a
rubber stamp.

Previous versions asked an LLM to derive assertion values from the user
scenario; that turned out to be unreliable because medoid-based action
sequences frequently don't satisfy canonical task-family success patterns,
and LLMs would anchor on the scenario rather than the observed state.
"""

import inspect
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.domain_config import DomainConfig
from src.common.domain_utils import WORKSPACE_ROOT, dict_to_task, ensure_tau2_path

ensure_tau2_path()
from tau2.data_model.tasks import EnvAssertion  # type: ignore
from tau2.domains.telecom.tools import TelecomTools  # type: ignore
from tau2.domains.telecom.user_tools import TelecomUserTools  # type: ignore


_LOGS_BASE = os.path.join(WORKSPACE_ROOT, "logs")

_MAX_ASSERTIONS = 4


class EnvAssertionSynthesisError(Exception):
    """Raised when synthesis fails (e.g. golden-action replay fails)."""


@dataclass
class _Failure:
    index: int
    func_name: str
    arguments: Dict[str, Any]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "func_name": self.func_name,
            "arguments": self.arguments,
            "reason": self.reason,
        }


_ENV_TYPE_CLASSES = {
    "assistant": TelecomTools,
    "user": TelecomUserTools,
}


class EnvAssertionSynthesizer:
    """Produce verified env_assertions for a telecom task.

    Telecom-only. Rule-based; no LLM call. Constructed once per generation
    run and called per task.
    """

    def __init__(
        self,
        domain_config: DomainConfig,
        llm_caller: Any = None,  # kept for call-site compat; no longer used
        max_retries: int = 3,    # kept for call-site compat; only one pass is attempted
    ):
        if domain_config.domain != "telecom":
            raise ValueError(
                f"EnvAssertionSynthesizer is telecom-only; got domain={domain_config.domain}"
            )
        self.domain_config = domain_config

    # -- public --------------------------------------------------------

    def synthesize(self, task_dict: Dict[str, Any]) -> List[EnvAssertion]:
        """Return a verified list of EnvAssertion objects for the task.

        Flow:
          1. Build fresh env, apply initial_state, replay golden actions.
          2. Capture end-state snapshot.
          3. Rule-based selector picks 1-4 assertions from the observed state.
          4. Verify on the same env (rubber-stamp).

        Raises EnvAssertionSynthesisError only on structural problems
        (golden actions fail to apply, or selected assertions unexpectedly
        fail to verify).
        """
        env, replay_error = self._build_env_and_replay(task_dict)
        if replay_error is not None:
            raise EnvAssertionSynthesisError(
                f"golden actions failed on gold env: {replay_error}"
            )

        end_state = self._capture_end_state(env)
        golden_actions = (
            task_dict.get("evaluation_criteria", {}).get("actions") or []
        )

        candidates = self._select_assertions(golden_actions, end_state)

        if not candidates:
            raise EnvAssertionSynthesisError(
                "no assertions could be derived from the end state"
            )

        assertions = self._instantiate(candidates)
        struct_failures = self._structural_check(assertions)
        if struct_failures:
            self._log_failure(task_dict, 1, struct_failures, "<programmatic>", end_state)
            raise EnvAssertionSynthesisError(
                f"structural check failed on programmatic assertions: {struct_failures}"
            )

        env_failures = self._verify_on_env(env, assertions)
        if env_failures:
            # This indicates a bug in the selector (snapshot → assertion mapping).
            # Log with full context so the bug can be diagnosed.
            self._log_failure(task_dict, 1, env_failures, "<programmatic>", end_state)
            raise EnvAssertionSynthesisError(
                f"programmatic assertions did not verify on gold env: {env_failures}"
            )

        return assertions

    def verify_existing(
        self, task_dict: Dict[str, Any], assertions: List[EnvAssertion]
    ) -> List[_Failure]:
        """Replay golden actions on a fresh env and run `assertions` against it.

        Used by adversarial evolution to confirm existing env_assertions still
        hold after the evolved initial_state. Returns a list of failures (empty
        list means all passed).
        """
        env, replay_error = self._build_env_and_replay(task_dict)
        if replay_error is not None:
            return [
                _Failure(
                    index=-1,
                    func_name="<replay>",
                    arguments={},
                    reason=replay_error,
                )
            ]
        return self._verify_on_env(env, assertions)

    # -- rule-based selector -------------------------------------------

    def _select_assertions(
        self, golden_actions: List[Dict[str, Any]], end_state: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Pick a prioritized subset of env_assertions from the observed state.

        Priority order (stop once _MAX_ASSERTIONS is reached):
          1. Write-action side-effects (each write action maps to an assertion
             that reads the exact end-state value its side-effect produced):
             - refuel_data -> assert_data_refueling_amount
             - suspend_line / resume_line -> assert_line_status
             - send_payment_request + make_payment -> assert_no_overdue_bill
          2. Goal-state assertions informative across most task families:
             - assert_service_status (always)
             - assert_mobile_data_status (if any user-side toggle that affects
               mobile data connectivity appears)
          3. Optional extras: airplane-mode, data-saver (only if the sequence
             modifies them and service_status already picked).
        """
        picks: List[Dict[str, Any]] = []
        assistant_state = end_state.get("assistant_state", {})
        user_state = end_state.get("user_state", {})
        derived = end_state.get("derived", {})
        lines_by_id = assistant_state.get("lines", {})
        bills_by_id = assistant_state.get("bills", {})

        action_names = {a.get("name") for a in golden_actions}
        # Index: last occurrence of each write action's args
        last_args_by_name: Dict[str, Dict[str, Any]] = {}
        for a in golden_actions:
            name = a.get("name")
            if name:
                last_args_by_name[name] = a.get("arguments") or {}

        # 1a. refuel_data -> assert_data_refueling_amount
        if "refuel_data" in last_args_by_name and len(picks) < _MAX_ASSERTIONS:
            args = last_args_by_name["refuel_data"]
            line_id = args.get("line_id")
            customer_id = args.get("customer_id")
            line = lines_by_id.get(line_id, {}) if line_id else {}
            observed = line.get("data_refueling_gb")
            if (
                isinstance(observed, (int, float))
                and observed > 0
                and line_id
                and customer_id
            ):
                picks.append({
                    "env_type": "assistant",
                    "func_name": "assert_data_refueling_amount",
                    "arguments": {
                        "customer_id": customer_id,
                        "line_id": line_id,
                        "expected_amount": float(observed),
                    },
                    "message": None,
                })

        # 1b. suspend_line / resume_line -> assert_line_status
        for name in ("suspend_line", "resume_line"):
            if name in last_args_by_name and len(picks) < _MAX_ASSERTIONS:
                args = last_args_by_name[name]
                line_id = args.get("line_id")
                customer_id = args.get("customer_id")
                line = lines_by_id.get(line_id, {}) if line_id else {}
                observed_status = line.get("status")
                if observed_status and line_id and customer_id:
                    picks.append({
                        "env_type": "assistant",
                        "func_name": "assert_line_status",
                        "arguments": {
                            "customer_id": customer_id,
                            "line_id": line_id,
                            "expected_status": observed_status,
                        },
                        "message": None,
                    })
                break  # one line_status assertion is enough

        # 1c. send_payment_request + make_payment -> assert_no_overdue_bill
        if (
            "send_payment_request" in last_args_by_name
            and "make_payment" in action_names
            and len(picks) < _MAX_ASSERTIONS
        ):
            bill_id = last_args_by_name["send_payment_request"].get("bill_id")
            bill = bills_by_id.get(bill_id, {}) if bill_id else {}
            observed_status = bill.get("status")
            # Only assert if the bill is actually paid/absent in end state
            if bill_id and (not bill or observed_status in ("Paid", "paid")):
                picks.append({
                    "env_type": "assistant",
                    "func_name": "assert_no_overdue_bill",
                    "arguments": {"overdue_bill_id": bill_id},
                    "message": None,
                })

        # 2a. assert_service_status (always informative)
        device = user_state.get("device", {}) or {}
        service_status_raw = device.get("network_connection_status")
        if service_status_raw and len(picks) < _MAX_ASSERTIONS:
            picks.append({
                "env_type": "user",
                "func_name": "assert_service_status",
                "arguments": {"expected_status": str(service_status_raw)},
                "message": None,
            })

        # 2b. assert_mobile_data_status (derived from snapshot)
        mobile_data_working = derived.get("mobile_data_working")
        data_affecting = {
            "toggle_data", "toggle_airplane_mode", "toggle_roaming",
            "toggle_data_saver_mode", "refuel_data", "reboot_device",
        }
        if (
            isinstance(mobile_data_working, bool)
            and data_affecting & action_names
            and len(picks) < _MAX_ASSERTIONS
        ):
            picks.append({
                "env_type": "user",
                "func_name": "assert_mobile_data_status",
                "arguments": {"expected_status": bool(mobile_data_working)},
                "message": None,
            })

        # 3. Optional extras
        if "toggle_airplane_mode" in action_names and len(picks) < _MAX_ASSERTIONS:
            airplane_mode = device.get("airplane_mode")
            if isinstance(airplane_mode, bool):
                picks.append({
                    "env_type": "user",
                    "func_name": "assert_airplane_mode_status",
                    "arguments": {"expected_status": bool(airplane_mode)},
                    "message": None,
                })

        if "toggle_data_saver_mode" in action_names and len(picks) < _MAX_ASSERTIONS:
            saver = device.get("data_saver_mode")
            if isinstance(saver, bool):
                picks.append({
                    "env_type": "user",
                    "func_name": "assert_mobile_data_saver_mode_status",
                    "arguments": {"expected_status": bool(saver)},
                    "message": None,
                })

        # Fallback: if we still have nothing, at least pick one assertion
        # from whatever state we have so the task ships with a reward target.
        if not picks:
            if service_status_raw:
                picks.append({
                    "env_type": "user",
                    "func_name": "assert_service_status",
                    "arguments": {"expected_status": str(service_status_raw)},
                    "message": None,
                })
            elif isinstance(mobile_data_working, bool):
                picks.append({
                    "env_type": "user",
                    "func_name": "assert_mobile_data_status",
                    "arguments": {"expected_status": bool(mobile_data_working)},
                    "message": None,
                })

        return picks

    # -- env setup and introspection -----------------------------------

    def _build_env_and_replay(self, task_dict: Dict[str, Any]):
        """Build a fresh env, apply initial_state, replay golden actions."""
        task = dict_to_task(task_dict)
        env = self.domain_config.get_environment(
            db=self.domain_config.get_db(),
            user_db=self.domain_config.get_user_db(),
        )
        init = task.initial_state
        env.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            message_history=[],
        )
        golden_actions = (
            task.evaluation_criteria.actions
            if task.evaluation_criteria and task.evaluation_criteria.actions
            else []
        )
        # Real simulation calls sync_tools after every tool call (orchestrator
        # line 361; set_state replay uses get_response which syncs internally).
        # We must match that behavior here — several telecom actions read
        # user_tools.db.surroundings values that sync_tools derives from
        # agent-side DB state (e.g., simulate_network_search consults
        # surroundings.line_active). Without per-action syncs, a later action
        # can observe stale surroundings and produce a different end state
        # than the real simulation, causing assertions to fail at eval time.
        for action in golden_actions:
            try:
                env.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
                env.sync_tools()
            except Exception as e:
                return env, f"action {action.name}({action.arguments}) raised: {e}"
        return env, None

    @staticmethod
    def _capture_end_state(env) -> Dict[str, Any]:
        """Extract a JSON-serializable snapshot of the env state."""
        def _model_dump(m):
            try:
                return m.model_dump(mode="json")
            except Exception:
                try:
                    return json.loads(m.model_dump_json())
                except Exception:
                    return json.loads(json.dumps(m, default=str))

        assistant_db = env.tools.db
        user_db = env.user_tools.db
        assistant_state = {
            "customers": {c.customer_id: _model_dump(c) for c in getattr(assistant_db, "customers", [])},
            "lines": {l.line_id: _model_dump(l) for l in getattr(assistant_db, "lines", [])},
            "bills": {b.bill_id: _model_dump(b) for b in getattr(assistant_db, "bills", [])},
        }
        user_state = {
            "device": _model_dump(user_db.device),
            "surroundings": _model_dump(user_db.surroundings),
        }
        derived: Dict[str, Any] = {}
        try:
            derived["mobile_data_working"] = env.user_tools._get_mobile_data_working()
        except Exception as e:
            derived["mobile_data_working"] = f"<error: {e}>"
        try:
            derived["can_send_mms"] = env.user_tools._can_send_mms()
        except Exception as e:
            derived["can_send_mms"] = f"<error: {e}>"
        try:
            speed, desc = env.user_tools._run_speed_test()
            derived["speed_test"] = [speed, desc]
        except Exception as e:
            derived["speed_test"] = f"<error: {e}>"
        return {
            "assistant_state": assistant_state,
            "user_state": user_state,
            "derived": derived,
        }

    # -- instantiation & checks ----------------------------------------

    @staticmethod
    def _instantiate(raw: List[Dict[str, Any]]) -> List[EnvAssertion]:
        out: List[EnvAssertion] = []
        for i, item in enumerate(raw):
            try:
                out.append(EnvAssertion(**item))
            except Exception as e:
                raise ValueError(
                    f"Assertion at index {i} could not be instantiated as EnvAssertion: {e}. Item: {item}"
                )
        return out

    def _structural_check(self, assertions: List[EnvAssertion]) -> List[_Failure]:
        failures: List[_Failure] = []
        for i, a in enumerate(assertions):
            cls = _ENV_TYPE_CLASSES.get(a.env_type)
            if cls is None:
                failures.append(
                    _Failure(i, a.func_name, a.arguments, f"unknown env_type {a.env_type!r}")
                )
                continue
            method = getattr(cls, a.func_name, None)
            if method is None or not callable(method):
                failures.append(
                    _Failure(i, a.func_name, a.arguments, f"{cls.__name__} has no method {a.func_name!r}")
                )
                continue
            sig = inspect.signature(method)
            params = [p for p in sig.parameters.values() if p.name != "self"]
            param_names = {p.name for p in params}
            required = {
                p.name for p in params if p.default is inspect.Parameter.empty
                and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            unknown = set(a.arguments.keys()) - param_names
            missing = required - set(a.arguments.keys())
            if unknown:
                failures.append(_Failure(i, a.func_name, a.arguments, f"unknown arguments {sorted(unknown)}"))
            if missing:
                failures.append(_Failure(i, a.func_name, a.arguments, f"missing required arguments {sorted(missing)}"))
        return failures

    @staticmethod
    def _verify_on_env(env, assertions: List[EnvAssertion]) -> List[_Failure]:
        failures: List[_Failure] = []
        for i, a in enumerate(assertions):
            try:
                ok = env.run_env_assertion(a, raise_assertion_error=False)
            except Exception as e:
                failures.append(_Failure(i, a.func_name, a.arguments, f"assertion raised: {e}"))
                continue
            if not ok:
                failures.append(_Failure(i, a.func_name, a.arguments, "did not hold on gold env"))
        return failures

    # -- logging -------------------------------------------------------

    def _log_failure(
        self,
        task_dict: Dict[str, Any],
        attempt: int,
        failures: List[_Failure],
        response: str,
        end_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        log_dir = os.path.join(_LOGS_BASE, "telecom", "env_assertion_errors")
        os.makedirs(log_dir, exist_ok=True)
        task_id = task_dict.get("id", "unknown")
        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(task_id))
        timestamp = int(time.time())
        path = os.path.join(log_dir, f"{safe_id}_{timestamp}_attempt{attempt}.json")
        payload = {
            "task_id": task_id,
            "attempt": attempt,
            "failures": [f.to_dict() for f in failures],
            "llm_response": response,
            "end_state": end_state,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
