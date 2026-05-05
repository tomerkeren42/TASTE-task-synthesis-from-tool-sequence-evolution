import json
from typing import Any, Dict, List, Optional, Tuple

from tool_spec_retriever import ToolsSpecRetriever
from call_to_llm import LLMCaller
from domain_utils import load_policy, load_tasks
from prompt_manager import PromptManager
from llm_response_parser import LLMResponseParser

EXAMPLE_TASK_IDS = [2, 7, 8, 17]


class TaskGenerator:
    """Generates tasks via LLM: creates user tasks and DB initialization."""

    def __init__(
        self,
        domain: str = "airline",
        model_name: str = "vertex_ai/gemini-3-flash-preview",
        max_output_tokens: int = 65536,
        save_prompts: bool = False,
        domain_config=None,
    ):
        self.domain = domain

        if domain_config is None:
            from src.common.domain_config import DomainConfig
            domain_config = DomainConfig(domain)

        tool_spec_retriever = ToolsSpecRetriever(path=domain_config.tool_spec_path)
        self.tool_spec_json = tool_spec_retriever.get_tool_spec_json()

        self.llm_caller = LLMCaller(
            model_name=model_name,
            max_output_tokens=max_output_tokens,
            save_prompts=save_prompts,
        )
        self.policy = load_policy(domain)
        self.prompt_manager = PromptManager(prompts_dir=domain_config.prompts_dir, domain=domain_config.domain)
        self.db_schema_str = domain_config.get_db_schema_str()

    # -- helpers --

    def _get_example_task_str(self) -> str:
        tasks_data = load_tasks(self.domain)
        example_tasks = [tasks_data[i] for i in EXAMPLE_TASK_IDS]
        return json.dumps(example_tasks, indent=2) if example_tasks else "No example available"

    def _get_evaluation_criteria_str(self, action_sequence: List[str]) -> str:
        actions = [
            {"action_id": str(idx), "name": name, "arguments": "...generate complete arguments..."}
            for idx, name in enumerate(action_sequence)
        ]
        return json.dumps(actions, indent=2)

    def _get_task_context_str(self, task_dict: Dict[str, Any]) -> str:
        return json.dumps({
            "user_scenario": task_dict.get("user_scenario", {}),
            "description": task_dict.get("description", {}),
        }, indent=2)

    # -- LLM generation steps --

    def create_user_task(self, action_sequence: List[str], error_context: str = "") -> Dict[str, Any]:
        """Use LLM to create a user task with scenario and action arguments."""
        prompt = self.prompt_manager.get_prompt("create_user_task").format(
            domain=self.domain,
            policy=self.policy,
            tool_spec=self.tool_spec_json,
            example_task=self._get_example_task_str(),
            evaluation_criteria=self._get_evaluation_criteria_str(action_sequence),
            action_sequence=action_sequence,
            error_context=error_context,
        )
        response = self.llm_caller.call(prompt=prompt, stage_name="create_user_task")
        task_dict = LLMResponseParser.extract_json(response)
        return LLMResponseParser.clean_task_dict(task_dict)

    def generate_db_initialization(self, task_dict: Dict[str, Any], error_context: str = "") -> Dict[str, Any]:
        """Use LLM to generate DB entities with correct relationships."""
        action_arguments = task_dict.get("evaluation_criteria", {}).get("actions", [])
        prompt = self.prompt_manager.get_prompt("generate_db_initialization").format(
            task_context=self._get_task_context_str(task_dict),
            action_arguments=json.dumps(action_arguments, indent=2, default=str),
            policy=self.policy,
            db_schema=self.db_schema_str,
            error_context=error_context,
        )
        response = self.llm_caller.call(prompt=prompt, stage_name="generate_db_initialization")
        return LLMResponseParser.extract_json(response)

    def patch_task(
        self, task_dict: Dict[str, Any], db_entities: Dict[str, Any], error: str
    ) -> Dict[str, Any]:
        """Fix a specific error in an existing task via LLM."""
        prompt = self.prompt_manager.get_prompt("patch_task").format(
            error=error,
            policy=self.policy,
            task_json=json.dumps(task_dict, indent=2, default=str),
            db_json=json.dumps(db_entities, indent=2, default=str),
        )
        response = self.llm_caller.call(prompt=prompt, stage_name="patch_task")
        patched = LLMResponseParser.extract_json(response)
        return LLMResponseParser.clean_task_dict(patched)

    def patch_db(
        self, task_dict: Dict[str, Any], db_entities: Dict[str, Any], error: str
    ) -> Dict[str, Any]:
        """Fix a specific error in an existing DB via LLM."""
        prompt = self.prompt_manager.get_prompt("patch_db").format(
            error=error,
            policy=self.policy,
            task_json=json.dumps(task_dict, indent=2, default=str),
            db_json=json.dumps(db_entities, indent=2, default=str),
        )
        response = self.llm_caller.call(prompt=prompt, stage_name="patch_db")
        return LLMResponseParser.extract_json(response)

    def generate(self, action_sequence: List[str], error_context: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate a complete task: user task + DB initialization (two LLM calls)."""
        task_dict = self.create_user_task(action_sequence, error_context)
        db_entities = self.generate_db_initialization(task_dict)
        return task_dict, db_entities

