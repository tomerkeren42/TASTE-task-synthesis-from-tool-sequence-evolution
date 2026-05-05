"""
Validates action sequences for viability using an LLM.

Checks that a sequence is realistic, pragmatic, policy-compliant,
environment-valid, and logically coherent.
"""
import json
from dataclasses import dataclass
from typing import List

from vertexai.generative_models import GenerationConfig

from src.common.call_to_llm import LLMCaller
from src.common.domain_utils import load_policy
from src.common.llm_response_parser import LLMResponseParser
from src.common.prompt_manager import PromptManager
from src.common.tool_spec_retriever import ToolsSpecRetriever


@dataclass
class ValidationResult:
    """Result of action sequence validation."""

    valid: bool
    reason: str
    problematic_indices: List[int]


class ActionSequenceValidator:
    """
    Validates whether an action sequence is viable for the benchmark.

    Uses an LLM to check that the sequence is:
    - Realistic (plausible customer service flow)
    - Pragmatic (correct ordering, prerequisites satisfied)
    - Policy-compliant (allowed under domain policy)
    - Environment-valid (valid tools, respects data model)
    - Logical (coherent cause-and-effect flow)
    """

    def __init__(
        self,
        domain: str = "airline",
        model_name: str = "vertex_ai/gemini-3-flash-preview",
        max_output_tokens: int = 16384,
        save_prompts: bool = False,
        domain_config=None,
        temperature: float = 0.0,
    ):
        if domain_config is None:
            from src.common.domain_config import DomainConfig
            domain_config = DomainConfig(domain)

        self.domain = domain
        self.policy = load_policy(domain)
        self.tool_spec_json = ToolsSpecRetriever(path=domain_config.tool_spec_path).get_tool_spec_json()
        self.prompt_manager = PromptManager(prompts_dir=domain_config.prompts_dir, domain=domain_config.domain)
        # Note: ``custom_config`` below is honored only on the Vertex AI path of
        # ``LLMCaller``; on the litellm path the temperature comes from the
        # ``LLMCaller`` constructor. Pass it explicitly so validation is
        # deterministic regardless of which provider is selected.
        self._generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.llm_caller = LLMCaller(
            model_name=model_name,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            save_prompts=save_prompts,
        )

    def validate(self, action_sequence: List[str]) -> ValidationResult:
        """
        Validate an action sequence for viability.

        Args:
            action_sequence: List of action names in order (e.g. ["get_reservation_details", "cancel_reservation"])

        Returns:
            ValidationResult with valid=True/False and reason string
        """
        if not action_sequence:
            return ValidationResult(
                valid=True,
                reason="Empty sequence is valid (conversation-only task).",
                problematic_indices=[],
            )

        prompt = self.prompt_manager.get_prompt("validate_action_sequence").format(
            action_sequence=json.dumps(action_sequence),
            policy=self.policy,
            tool_spec=self.tool_spec_json,
        )

        response = self.llm_caller.call(
            prompt=prompt,
            stage_name="validate_action_sequence",
            custom_config=self._generation_config,
        )

        parsed = LLMResponseParser.extract_json(response)
        valid = parsed.get("valid", False)
        reason = parsed.get("reason", "No reason provided.")
        raw_indices = parsed.get("problematic_indices", [])
        # Sanitise: keep only valid integer indices within range
        seq_len = len(action_sequence)
        problematic_indices = [
            int(i) for i in raw_indices
            if isinstance(i, (int, float)) and 0 <= int(i) < seq_len
        ]

        return ValidationResult(valid=valid, reason=reason, problematic_indices=problematic_indices)
