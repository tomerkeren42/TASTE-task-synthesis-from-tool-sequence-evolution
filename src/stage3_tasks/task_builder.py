from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from src.common.domain_utils import ensure_tau2_path

ensure_tau2_path()
from tau2.data_model.tasks import (
    Task,
    Description,
    UserScenario,
    StructuredUserInstructions,
    InitialState,
    InitializationData,
    EvaluationCriteria,
    Action,
    EnvAssertion,
    RewardType,
)


@dataclass
class ValidationMetadata:
    """Metadata about the task generation/validation process."""
    
    success: bool = False
    solver_retries: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, including only set optional fields."""
        result: Dict[str, Any] = {
            "validation_success": self.success,
            "solver_retries": self.solver_retries,
        }
        if self.error is not None:
            result["validation_error"] = self.error
        return result


@dataclass
class GeneratedTask:
    """
    Wrapper combining a tau2 Task with validation metadata.
    
    Use GeneratedTask.create() to build from LLM output.
    """
    
    task: Task
    validation: ValidationMetadata = field(default_factory=ValidationMetadata)

    @classmethod
    def create(
        cls,
        task_dict: Dict[str, Any],
        db_entities: Optional[Dict[str, Any]] = None,
        domain: str = "airline",
    ) -> "GeneratedTask":
        """
        Create a GeneratedTask from LLM output and optional DB entities.
        
        Args:
            task_dict: Task dictionary from create_user_task containing:
                       - description (with purpose, relevant_policies, notes)
                       - user_scenario (with persona, instructions)
                       - evaluation_criteria (with actions)
            db_entities: Optional DB entities to initialize (users, reservations, flights).
                        These will be included in initial_state.initialization_data.agent_data.
            
        Returns:
            GeneratedTask instance wrapping a tau2 Task
        """
        # Build Description
        desc_data = task_dict.get("description", {})
        description = Description(
            purpose=desc_data.get("purpose"),
            relevant_policies=desc_data.get("relevant_policies"),
            notes=desc_data.get("notes"),
        ) if desc_data else None
        
        # Build UserScenario
        scenario_data = task_dict.get("user_scenario", {})
        instructions_data = scenario_data.get("instructions", {})
        
        if isinstance(instructions_data, dict):
            instructions = StructuredUserInstructions(
                domain=instructions_data.get("domain", "airline"),
                reason_for_call=instructions_data.get("reason_for_call", ""),
                known_info=instructions_data.get("known_info"),
                unknown_info=instructions_data.get("unknown_info"),
                task_instructions=instructions_data.get("task_instructions", ""),
            )
        else:
            # String instructions
            instructions = instructions_data
        
        user_scenario = UserScenario(
            persona=scenario_data.get("persona"),
            instructions=instructions,
        )
        
        # Build InitialState with DB entities
        initial_state = None
        has_entities = db_entities and any(
            db_entities.get(key) for key in [
                # airline
                "users", "reservations", "flights",
                # retail
                "orders", "products",
                # telecom
                "customers", "plans", "lines", "bills", "devices",
                "device", "surroundings",
            ]
        )
        if has_entities:
            agent_data = dict(db_entities)
            user_data = None
            if domain == "telecom":
                # TelecomDB expects List fields; LLM outputs dict-of-dicts. Convert
                # so the saved task is directly loadable by tau2's set_state.
                # Also split device/surroundings into user_data (TelecomUserDB).
                from src.common.domain_validators.telecom import TelecomValidator
                for entity_type in ("customers", "plans", "lines", "bills", "devices"):
                    if entity_type in agent_data:
                        agent_data[entity_type] = TelecomValidator._dict_to_list(
                            entity_type, agent_data[entity_type]
                        )
                user_data = {}
                for key in ("device", "surroundings"):
                    if key in agent_data:
                        user_data[key] = agent_data.pop(key)
                if not user_data:
                    user_data = None
            initialization_data = InitializationData(
                agent_data=agent_data,
                user_data=user_data,
            )
            initial_state = InitialState(
                initialization_data=initialization_data,
                initialization_actions=None,
                message_history=None,
            )
        
        # Build EvaluationCriteria
        eval_data = task_dict.get("evaluation_criteria", {})
        actions_data = eval_data.get("actions", [])
        actions = [
            Action(
                action_id=str(action.get("action_id", idx)),
                name=action.get("name", ""),
                arguments=action.get("arguments", {}),
                requestor=action.get("requestor", "assistant"),
                info=action.get("info") if isinstance(action.get("info"), str) else None,
                compare_args=action.get("compare_args"),
            )
            for idx, action in enumerate(actions_data)
        ]
        
        # env_assertions and reward_basis may already have been populated by the
        # telecom synthesizer mutating the caller's task_dict. Preserve what's
        # there so create()-after-validation keeps the synthesized assertions.
        raw_env_assertions = eval_data.get("env_assertions")
        env_assertions = None
        if raw_env_assertions:
            env_assertions = [
                a if isinstance(a, EnvAssertion) else EnvAssertion(**a)
                for a in raw_env_assertions
            ]

        raw_reward_basis = eval_data.get("reward_basis")
        if raw_reward_basis:
            reward_basis = [
                r if isinstance(r, RewardType) else RewardType(r)
                for r in raw_reward_basis
            ]
        else:
            reward_basis = [RewardType.DB, RewardType.COMMUNICATE]

        evaluation_criteria = EvaluationCriteria(
            actions=actions,
            env_assertions=env_assertions,
            # Airline-only: preserve communicate_info if a caller populated it
            # upstream so it survives into tasks.json. Other domains never set
            # this key, and we explicitly guard on domain to avoid scoring them
            # on tau2's COMMUNICATE criterion.
            communicate_info=(
                eval_data.get("communicate_info") if domain == "airline" else None
            ),
            nl_assertions=None,
            reward_basis=reward_basis,
        )
        
        # Create tau2 Task
        task = Task(
            id="generated_unassigned",
            description=description,
            user_scenario=user_scenario,
            ticket=None,
            initial_state=initial_state,
            evaluation_criteria=evaluation_criteria,
        )
        
        return cls(task=task)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for serialization/logging.
        
        Merges the tau2 Task dict with validation metadata.
        """
        result = self.task.model_dump(mode="json")
        result.update(self.validation.to_dict())
        return result
