import json
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

from src.common.domain_config import DomainConfig
from src.common.domain_utils import dict_to_task, ensure_tau2_path, load_policy
from src.stage3_tasks.database_state_manager import DatabaseStateManager
from src.stage3_tasks.action_sequence_policy import apply_policy_completion
from src.stage3_tasks.env_assertion_synthesizer import (
    EnvAssertionSynthesizer,
    EnvAssertionSynthesisError,
)
from src.stage3_tasks.scenario_aligner import ScenarioAligner
from src.stage3_tasks.task_validator_logging import (
    log_solver_error,
    log_gt_agent_error,
    log_template_error,
    log_db_preflight_error,
    log_db_schema_error,
    log_coherence_error,
)
from src.stage3_tasks.task_builder import GeneratedTask
from src.common.call_to_llm import LLMCaller
from src.common.prompt_manager import PromptManager
from src.common.llm_response_parser import LLMResponseParser
from src.stage3_tasks.partial_coverage_gt_agent import PartialCoverageGTAgent

ensure_tau2_path()
from tau2.run import run_task  # type: ignore
from tau2.data_model.message import ToolCall  # type: ignore


@dataclass
class ValidationResult:
    success: bool
    db_ok: bool
    gt_agent_success: bool
    error: Optional[str] = None
    failed_step: Optional[str] = None
    # Populated only on GT agent success, carries messages + reward info.
    # Typed as Any to avoid pulling the tau2 simulation module into this
    # file's import graph at module load.
    gt_simulation: Optional[Any] = None


class RuleBasedValidator:
    """Validates task structure, DB consistency, and action executability via deterministic checks.

    Runs in order: template check → DB preflight → DB schema → solver → coherence (optional).
    Short-circuits on first failure.
    """

    def __init__(
        self,
        domain: str = "airline",
        coherence_llm: Optional[str] = None,
        domain_config: Optional[DomainConfig] = None,
    ):
        self.domain = domain
        self.coherence_llm = coherence_llm

        if domain_config is None:
            domain_config = DomainConfig(domain)
        self.domain_validator = domain_config.get_domain_validator()
        self.db_state_manager = DatabaseStateManager(
            self.domain_validator.get_empty_db_dict()
        )
        self._policy = load_policy(domain)
        self._prompt_manager = PromptManager(prompts_dir=domain_config.prompts_dir, domain=domain_config.domain)
        # Cache tool parameter names per action for deterministic arg-shape
        # validation. Catches LLM hallucinations (e.g. passing `amount` to
        # `send_payment_request`) in O(1) before the solver simulation runs.
        from src.common.tool_spec_retriever import ToolsSpecRetriever
        self._action_params = ToolsSpecRetriever(
            path=domain_config.tool_spec_path
        ).get_action_params()

    def validate(self, task_dict: Dict[str, Any], db_entities: Dict[str, Any]) -> ValidationResult:
        """Run all rule-based validations sequentially. Short-circuits on first failure."""
        action_arguments = task_dict.get("evaluation_criteria", {}).get("actions", [])

        if not action_arguments:
            return ValidationResult(success=True, db_ok=True, gt_agent_success=False)

        template_errors = self._check_template_variables(action_arguments)
        if template_errors:
            error_msg = "Task has unresolved template variables: " + "; ".join(template_errors)
            print(f"  -> {error_msg}")
            log_template_error(task_dict, db_entities, template_errors, domain=self.domain)
            return ValidationResult(
                success=False, db_ok=True, gt_agent_success=False,
                error=error_msg, failed_step="template_check",
            )

        arg_shape_errors = self._check_arg_shapes(action_arguments)
        if arg_shape_errors:
            error_msg = "Action argument shape check failed: " + "; ".join(arg_shape_errors)
            print(f"  -> {error_msg}")
            return ValidationResult(
                success=False, db_ok=True, gt_agent_success=False,
                error=error_msg, failed_step="db_preflight",
            )

        preflight_errors = self._check_db_preflight(action_arguments, db_entities)
        if preflight_errors:
            error_msg = "DB preflight check failed: " + "; ".join(preflight_errors)
            print(f"  -> {error_msg}")
            log_db_preflight_error(task_dict, db_entities, preflight_errors, domain=self.domain)
            return ValidationResult(
                success=False, db_ok=False, gt_agent_success=False,
                error=error_msg, failed_step="db_preflight",
            )

        schema_errors = self._validate_db_schema(db_entities)
        if schema_errors:
            error_msg = "DB schema validation failed:\n" + "\n".join(f"  - {e}" for e in schema_errors)
            print(f"  -> {error_msg}")
            log_db_schema_error(task_dict, db_entities, schema_errors, domain=self.domain)
            return ValidationResult(
                success=False, db_ok=False, gt_agent_success=False,
                error=error_msg, failed_step="db_schema",
            )

        initial_db = self.db_state_manager.apply_entities(db_entities)
        result = self._run_solver_validation(action_arguments, initial_db, task_dict)
        if not result.success:
            return result

        if self.coherence_llm:
            coherence_result = self._run_coherence_review(task_dict, db_entities)
            if not coherence_result.success:
                return coherence_result

        return ValidationResult(success=True, db_ok=True, gt_agent_success=False)

    # ------------------------------------------------------------------
    # Template variable check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_template_variables(action_arguments: List[Dict[str, Any]]) -> List[str]:
        """Detect unresolved template variables like {action_3_reservation_id} in arguments."""
        import re
        errors: List[str] = []
        pattern = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")

        for action in action_arguments:
            name = action.get("name", "")
            args = action.get("arguments", {})
            args_str = json.dumps(args)
            matches = pattern.findall(args_str)
            if matches:
                errors.append(
                    f"Action '{name}': arguments contain unresolved placeholders: {matches}. "
                    f"All arguments must have concrete values, not template variables."
                )
        return errors

    # ------------------------------------------------------------------
    # Arg-shape check (deterministic, pre-solver)
    # ------------------------------------------------------------------

    def _check_arg_shapes(
        self, action_arguments: List[Dict[str, Any]]
    ) -> List[str]:
        """Reject actions whose argument keys don't match the tool spec.

        Catches the common LLM hallucination where extra kwargs are added
        (e.g. `amount` to `send_payment_request`). Running the solver would
        also catch these but costs a full env simulation; this is O(N) on
        the action list.

        Returns a list of human-readable errors, empty list if all pass.
        The failed_step is deliberately reported as "db_preflight" in the
        caller so `_get_patch_type` routes this to task_and_db patching
        (see validation_with_retry.py::_get_patch_type — it already special-
        cases "unexpected argument(s)" and "missing required argument" to
        trigger task-level patches).
        """
        errors: List[str] = []
        for action in action_arguments:
            name = action.get("name", "")
            spec_params = self._action_params.get(name)
            if spec_params is None:
                # Unknown tool entirely — caught elsewhere (solver) but flag
                # it here so the patch LLM sees it immediately.
                errors.append(
                    f"Action '{name}': unknown tool name (not in tool spec)."
                )
                continue
            args = action.get("arguments") or {}
            if not isinstance(args, dict):
                errors.append(
                    f"Action '{name}': arguments must be an object, got {type(args).__name__}"
                )
                continue
            allowed = set(spec_params.keys())
            unexpected = set(args.keys()) - allowed
            if unexpected:
                errors.append(
                    f"Action '{name}': unexpected argument(s) {sorted(unexpected)}. "
                    f"Allowed parameters: {sorted(allowed) or '(none)'}"
                )
        return errors

    # ------------------------------------------------------------------
    # DB preflight check
    # ------------------------------------------------------------------

    def _check_db_preflight(
        self,
        action_arguments: List[Dict[str, Any]],
        db_entities: Dict[str, Any],
    ) -> List[str]:
        """Delegate to domain-specific preflight checks."""
        return self.domain_validator.check_preflight(action_arguments, db_entities)

    # ------------------------------------------------------------------
    # DB schema validation (catches issues before Pydantic union explosion)
    # ------------------------------------------------------------------

    def _validate_db_schema(self, db_entities: Dict[str, Any]) -> List[str]:
        """Delegate to domain-specific schema validation."""
        return self.domain_validator.validate_db_schema(db_entities)

    # ------------------------------------------------------------------
    # Solver validation
    # ------------------------------------------------------------------

    def _execute_action(
        self,
        action_name: str,
        arguments: Dict[str, Any],
        db_state: Dict[str, Any],
        requestor: str = "assistant",
    ) -> Dict[str, Any]:
        """Execute a single action on the database state."""
        db, env = self.domain_validator.build_environment(db_state)
        tool_call = ToolCall(
            id="",
            name=action_name,
            arguments=arguments,
            requestor=requestor,
        )
        tool_message = env.get_response(tool_call)

        if hasattr(tool_message, "error") and tool_message.error:
            error_content = (
                tool_message.content
                if hasattr(tool_message, "content") and tool_message.content
                else str(tool_message)
            )
            if isinstance(error_content, str) and error_content.startswith("Error: "):
                error_message = error_content[7:]
            else:
                error_message = str(error_content) if error_content else "Unknown error"
            raise ValueError(f"Tool execution error for action '{action_name}': {error_message}")

        if env.tools is None:
            raise ValueError("Environment tools not available")

        updated_db = env.tools.db
        if hasattr(updated_db, "model_dump"):
            result = updated_db.model_dump()
        elif hasattr(updated_db, "dict"):
            result = updated_db.dict()
        else:
            result = json.loads(updated_db.json())

        # Preserve user DB state if present (telecom dual-DB)
        if hasattr(env, "user_tools") and env.user_tools is not None:
            user_db = env.user_tools.db
            if hasattr(user_db, "model_dump"):
                user_dict = user_db.model_dump()
            elif hasattr(user_db, "dict"):
                user_dict = user_db.dict()
            else:
                user_dict = json.loads(user_db.json())
            result.update(user_dict)

        return result

    def _solve(
        self,
        initial_db: Dict[str, Any],
        action_arguments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the action sequence on the initial DB state sequentially."""
        current_db_state = json.loads(json.dumps(initial_db, default=str))

        for action_arg in action_arguments:
            action_name = action_arg.get("name", "")
            action_args = action_arg.get("arguments", {})
            requestor = action_arg.get("requestor", "assistant")

            try:
                current_db_state = self._execute_action(
                    action_name, action_args, current_db_state, requestor=requestor
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Action '{action_name}' failed: {str(e)}",
                }

        return {"success": True, "error": None}

    def _run_solver_validation(
        self,
        action_arguments: List[Dict[str, Any]],
        initial_db: Dict[str, Any],
        task_dict: Dict[str, Any],
    ) -> ValidationResult:
        print("  -> Running solver validation...")
        result = self._solve(initial_db=initial_db, action_arguments=action_arguments)
        if not result["success"]:
            log_solver_error(action_arguments=action_arguments, error=result["error"], task_dict=task_dict, domain=self.domain)
            return ValidationResult(
                success=False, db_ok=False, gt_agent_success=False,
                error=f"Solver validation failed: {result['error']}", failed_step="solver")
        return ValidationResult(success=True, db_ok=True, gt_agent_success=False)

    # ------------------------------------------------------------------
    # LLM-based coherence review (optional)
    # ------------------------------------------------------------------

    def _run_coherence_review(
        self, task_dict: Dict[str, Any], db_entities: Dict[str, Any]
    ) -> ValidationResult:
        """Run LLM-based coherence review to catch policy violations and scenario issues."""
        print("  -> Running coherence review...")

        task_json = {
            "description": task_dict.get("description"),
            "user_scenario": task_dict.get("user_scenario"),
            "evaluation_criteria": task_dict.get("evaluation_criteria"),
        }

        prompt = self._prompt_manager.get_prompt("task_coherence_review").format(
            domain=self.domain,
            policy=self._policy,
            task_json=json.dumps(task_json, indent=2, default=str),
            db_json=json.dumps(db_entities, indent=2, default=str),
        )

        caller = LLMCaller(model_name=self.coherence_llm, max_output_tokens=32768)

        try:
            response = caller.call(prompt=prompt, stage_name="coherence_review")
            result = LLMResponseParser.extract_json(response)
        except Exception as e:
            # Don't block the pipeline on coherence review errors
            print(f"  -> Coherence review error (skipping): {e}")
            return ValidationResult(success=True, db_ok=True, gt_agent_success=False)

        verdict = result.get("verdict", "PASS")
        issues = result.get("issues", [])
        critical = [i for i in issues if i.get("severity") == "CRITICAL"]

        if verdict == "FAIL" and critical:
            descriptions = "; ".join(
                f"[{c.get('category', '?')}] {c.get('description', '?')}"
                for c in critical
            )
            error_msg = f"Coherence review failed: {descriptions}"
            print(f"  -> {error_msg}")
            log_coherence_error(task_dict, db_entities, error_msg, review_response=response, domain=self.domain)
            return ValidationResult(
                success=False, db_ok=True, gt_agent_success=False,
                error=error_msg, failed_step="coherence",
            )

        warnings = [i for i in issues if i.get("severity") == "WARNING"]
        if warnings:
            print(f"  -> Coherence review passed with {len(warnings)} warning(s)")

        return ValidationResult(success=True, db_ok=True, gt_agent_success=False)


class GTAgentValidator:
    """Validates tasks by running the GT (ground truth) agent simulation.

    Runs the golden action sequence through the partial-coverage GT agent
    once and reports the outcome.
    """

    def __init__(
        self,
        domain: str = "airline",
        gt_llm: str = "vertex_ai/gemini-3-flash-preview",
        partial_coverage_p: float = 0.33,
        gt_coverage_shuffle: bool = True,
        adversarial_user: bool = False,
        write_only: bool = False,
    ):
        self.domain = domain
        self.gt_llm = gt_llm
        self.partial_coverage_p = partial_coverage_p
        self.gt_coverage_shuffle = gt_coverage_shuffle
        self.adversarial_user = adversarial_user
        self.write_only = write_only

    @staticmethod
    def _convert_telecom_agent_data(task_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert telecom agent_data dict-of-dicts to lists for TelecomDB.

        TelecomDB expects List fields but our pipeline stores entities as
        dict-of-dicts.  Also splits user DB fields (device, surroundings)
        into user_data so tau2 can construct TelecomUserDB separately.
        """
        from src.common.domain_validators.telecom import TelecomValidator
        init_data = (
            task_dict.get("initial_state", {})
            .get("initialization_data", {})
        )
        if not init_data:
            return task_dict

        agent_data = init_data.get("agent_data")
        if not agent_data:
            return task_dict

        import copy
        task_dict = copy.deepcopy(task_dict)
        agent_data = task_dict["initial_state"]["initialization_data"]["agent_data"]

        # Convert dict-of-dicts to lists for TelecomDB entity types
        for entity_type in ("customers", "lines", "bills", "devices", "plans"):
            if entity_type in agent_data:
                agent_data[entity_type] = TelecomValidator._dict_to_list(
                    entity_type, agent_data[entity_type]
                )

        # Move user DB fields (device, surroundings) to user_data
        user_data = {}
        for key in ("device", "surroundings"):
            if key in agent_data:
                user_data[key] = agent_data.pop(key)
        if user_data:
            task_dict["initial_state"]["initialization_data"]["user_data"] = user_data

        return task_dict

    _telecom_domain_patched = False

    @classmethod
    def _patch_telecom_domain_in_registry(cls):
        """Replace the telecom domain constructor so the GT agent LLM's
        hallucinated user-tool calls don't crash with 'Tool not found'.

        The GT agent LLM (e.g. Gemini) sometimes calls user tools directly
        (requestor='assistant') instead of instructing the user to call them.
        This override makes the environment fall back to user_tools when a
        tool is not found in agent tools, mirroring the existing solo_mode
        behaviour in Environment.make_tool_call.
        """
        if cls._telecom_domain_patched:
            return

        from tau2.domains.telecom.environment import TelecomEnvironment

        original_make_tool_call = TelecomEnvironment.make_tool_call

        def _make_tool_call_with_fallback(self, tool_name, requestor="assistant", **kwargs):
            """Fallback: if assistant requests a tool only in user_tools, route there."""
            if (
                requestor == "assistant"
                and not self.solo_mode
                and self.user_tools is not None
                and not self.tools.has_tool(tool_name)
                and self.user_tools.has_tool(tool_name)
            ):
                return self.use_user_tool(tool_name=tool_name, **kwargs)
            return original_make_tool_call(self, tool_name, requestor=requestor, **kwargs)

        TelecomEnvironment.make_tool_call = _make_tool_call_with_fallback
        cls._telecom_domain_patched = True

    def validate(self, full_task_dict: Dict[str, Any]) -> ValidationResult:
        """Run GT agent validation on a fully assembled task dict (initial_state embedded).

        Runs the partial-coverage GT agent (with configurable shuffle) once
        and returns the outcome.
        """
        print(
            f"  -> Running GT agent validation ("
            f"partial_coverage p={self.partial_coverage_p}, "
            f"shuffle={self.gt_coverage_shuffle}, "
            f"write_only={self.write_only})..."
        )

        if self.domain == "telecom":
            full_task_dict = self._convert_telecom_agent_data(full_task_dict)
            self._patch_telecom_domain_in_registry()

        task = dict_to_task(full_task_dict)

        from tau2.registry import registry as tau2_registry

        adv = self.adversarial_user
        p = self.partial_coverage_p
        shuffle = self.gt_coverage_shuffle
        wo = self.write_only
        agent_key = (
            f"llm_agent_gt_partial_coverage_p{p}_s{int(shuffle)}"
            + ("_adv" if adv else "")
            + ("_wo" if wo else "")
        )
        try:
            class _PartialCoverageGTAgent(PartialCoverageGTAgent):
                def __init__(self_, tools, domain_policy, task, llm=None, llm_args=None, provide_function_args=True):
                    super().__init__(
                        tools, domain_policy, task, llm, llm_args, provide_function_args,
                        coverage_p=p, shuffle=shuffle,
                        adversarial_user=adv, write_only=wo,
                    )
            tau2_registry.register_agent(_PartialCoverageGTAgent, agent_key)
        except ValueError:
            pass

        return self._run_single_gt_agent(task, full_task_dict, agent_key, "single attempt")

    def _run_single_gt_agent(
        self,
        task: Any,
        task_dict: Dict[str, Any],
        agent_name: str,
        label: str,
    ) -> ValidationResult:
        """Run one GT agent pass and return the result."""
        try:
            simulation = run_task(
                domain=self.domain, task=task,
                agent=agent_name, user="user_simulator",
                llm_agent=self.gt_llm, llm_user=self.gt_llm,
            )
        except Exception as e:
            error = f"GT agent ({label}) simulation crashed: {str(e)}"
            print(f"  -> {error}")
            return ValidationResult(
                success=False, db_ok=True, gt_agent_success=False,
                error=error, failed_step="gt_agent",
            )

        reward_info = simulation.reward_info
        db_reward = reward_info.db_check.db_reward if reward_info.db_check else None
        overall_reward = reward_info.reward
        action_rewards = (
            [ac.action_reward for ac in reward_info.action_checks]
            if reward_info.action_checks else []
        )

        # Use overall reward (respects task's reward_basis) instead of
        # db_reward alone. Telecom uses ENV_ASSERTION-based reward, not DB.
        if overall_reward == 1.0:
            return ValidationResult(
                success=True,
                db_ok=True,
                gt_agent_success=True,
                gt_simulation=simulation,
            )

        error = f"GT agent ({label}): reward is {overall_reward}, expected 1.0 (db_reward={db_reward})"
        log_gt_agent_error(
            task_dict=task_dict, simulation=simulation,
            db_reward=db_reward, action_rewards=action_rewards,
            error=error, reward_info=reward_info, gt_llm=self.gt_llm,
            domain=self.domain,
        )
        return ValidationResult(
            success=False, db_ok=True, gt_agent_success=False,
            error=f"GT agent validation failed: {error}",
            failed_step="gt_agent",
        )


class TaskValidator:
    """Orchestrates RuleBasedValidator and GTAgentValidator.

    Maintains the same public interface as before for backward compatibility.
    """

    def __init__(
        self,
        domain: str = "airline",
        gt_llm: str = "vertex_ai/gemini-3-flash-preview",
        coherence_llm: Optional[str] = None,
        partial_coverage_p: float = 0.33,
        gt_coverage_shuffle: bool = True,
        adversarial_user: bool = False,
        domain_config: Optional[DomainConfig] = None,
        write_only: bool = False,
    ):
        self.domain = domain
        self.write_only = write_only
        if domain_config is None:
            domain_config = DomainConfig(domain)
        self._domain_config = domain_config
        self._rule_validator = RuleBasedValidator(
            domain=domain,
            coherence_llm=coherence_llm,
            domain_config=domain_config,
        )
        self._gt_validator = GTAgentValidator(
            domain=domain,
            gt_llm=gt_llm,
            partial_coverage_p=partial_coverage_p,
            gt_coverage_shuffle=gt_coverage_shuffle,
            adversarial_user=adversarial_user,
            write_only=write_only,
        )
        if domain == "telecom":
            self._env_assertion_synthesizer = EnvAssertionSynthesizer(
                domain_config=domain_config,
                llm_caller=LLMCaller(model_name=gt_llm, max_output_tokens=8192),
            )
            # Scenario aligner runs AFTER env_assertion synthesis and rewrites
            # user_scenario so the user-LLM drives toward the end state the
            # goldens actually produce. Never-raises; on failure the original
            # scenario is kept and GT still gets a shot.
            self._scenario_aligner = ScenarioAligner(
                domain_config=domain_config,
                llm_caller=LLMCaller(model_name=gt_llm, max_output_tokens=8192),
            )
        else:
            self._env_assertion_synthesizer = None
            self._scenario_aligner = None

    def validate_task(self, task_dict: Dict[str, Any], db_entities: Dict[str, Any]) -> ValidationResult:
        """Run rule-based validation, then env_assertion synthesis (telecom only),
        then GT agent validation. Short-circuits on first failure.
        """
        action_arguments = task_dict.get("evaluation_criteria", {}).get("actions", [])
        if not action_arguments:
            return ValidationResult(success=True, db_ok=True, gt_agent_success=True)

        patched_actions, policy_notes = apply_policy_completion(action_arguments, self.domain)
        if policy_notes:
            print(f"  -> policy-completion: {'; '.join(policy_notes)}")
            task_dict["evaluation_criteria"]["actions"] = patched_actions
            action_arguments = patched_actions

        result = self._rule_validator.validate(task_dict, db_entities)
        if not result.success:
            return result

        complete_task = GeneratedTask.create(task_dict, db_entities, domain=self.domain).to_dict()

        if self._env_assertion_synthesizer is not None:
            # Pass both dicts: the synthesizer replays from complete_task
            # (it has embedded initial_state) but MUST also mutate task_dict
            # so the synthesized env_assertions + reward_basis persist to
            # the saved tasks.json. Without this, the saved tasks have
            # reward_basis defaulting to ["DB","COMMUNICATE"] and empty
            # env_assertions, so downstream tau2 simulations score them
            # against DB-match (always fails for stochastic agents) and
            # give reward=0.
            synth_result = self._synthesize_env_assertions(complete_task, task_dict)
            if not synth_result.success:
                return synth_result

            # Align the scenario with the now-known end state. Mutates
            # task_dict (and complete_task) on success; never-raises. If
            # alignment fails (or the scenario already matches), GT runs
            # against whatever scenario is on the task.
            if self._scenario_aligner is not None:
                try:
                    aligned = self._scenario_aligner.align(task_dict)
                except Exception as e:
                    print(f"  -> scenario alignment raised unexpectedly: {e}")
                    aligned = False
                if aligned:
                    # Propagate the refined user_scenario into complete_task
                    # so the GT run below uses the new instructions.
                    complete_task["user_scenario"] = task_dict["user_scenario"]
                    if task_dict.get("description"):
                        complete_task["description"] = task_dict["description"]

        return self._gt_validator.validate(complete_task)

    def validate_gt_only(
        self,
        full_task_dict: Dict[str, Any],
        verify_env_assertions: bool = False,
    ) -> ValidationResult:
        """Run GT agent validation only -- skip rule-based checks.

        Use when the task's action sequence and initial_state are already valid
        (e.g., after style-only evolution where only user_scenario changed).

        Args:
            full_task_dict: Complete task dict with initial_state embedded,
                            as serialized by TaskSetManager (tasks.json format).
            verify_env_assertions: If True and the task has env_assertions, re-run
                            them on a fresh gold env after the (possibly evolved)
                            initial_state to confirm the evolution did not break
                            the assertions. Telecom-only effect.
        """
        if (
            verify_env_assertions
            and self._env_assertion_synthesizer is not None
            and full_task_dict.get("evaluation_criteria", {}).get("env_assertions")
        ):
            check_result = self._verify_existing_env_assertions(full_task_dict)
            if not check_result.success:
                return check_result
        return self._gt_validator.validate(full_task_dict)

    def _synthesize_env_assertions(
        self,
        complete_task: Dict[str, Any],
        task_dict: Dict[str, Any],
    ) -> ValidationResult:
        """Synthesize env_assertions and mutate both the complete_task (used
        for downstream GT validation in this call) AND the caller's task_dict
        (what gets saved into tasks.json).

        On success, populates env_assertions and sets reward_basis=[ENV_ASSERTION]
        on BOTH dicts so the fields survive into tasks.json.
        """
        try:
            assertions = self._env_assertion_synthesizer.synthesize(complete_task)
        except EnvAssertionSynthesisError as e:
            print(f"  -> env_assertion synthesis failed: {e}")
            return ValidationResult(
                success=False, db_ok=True, gt_agent_success=False,
                error=str(e), failed_step="env_assertion_synthesis",
            )

        serialized = [json.loads(a.model_dump_json()) for a in assertions]
        for dct in (complete_task, task_dict):
            eval_criteria = dct.setdefault("evaluation_criteria", {})
            eval_criteria["env_assertions"] = serialized
            eval_criteria["reward_basis"] = ["ENV_ASSERTION"]
        return ValidationResult(success=True, db_ok=True, gt_agent_success=False)

    def _verify_existing_env_assertions(
        self, task_dict: Dict[str, Any]
    ) -> ValidationResult:
        """Re-run existing env_assertions on a fresh gold env. Returns failure if any do not hold."""
        synth = self._env_assertion_synthesizer
        existing_raw = task_dict.get("evaluation_criteria", {}).get("env_assertions") or []
        from tau2.data_model.tasks import EnvAssertion as _EnvAssertion  # type: ignore
        try:
            assertions = [_EnvAssertion(**a) for a in existing_raw]
        except Exception as e:
            return ValidationResult(
                success=False, db_ok=True, gt_agent_success=False,
                error=f"existing env_assertions malformed: {e}",
                failed_step="env_assertion_verify",
            )
        failures = synth.verify_existing(task_dict, assertions)
        if failures:
            error = "evolved task broke env_assertions: " + "; ".join(
                f"[{f.index}] {f.func_name} {f.reason}" for f in failures
            )
            print(f"  -> {error}")
            return ValidationResult(
                success=False, db_ok=True, gt_agent_success=False,
                error=error, failed_step="env_assertion_verify",
            )
        return ValidationResult(success=True, db_ok=True, gt_agent_success=False)
