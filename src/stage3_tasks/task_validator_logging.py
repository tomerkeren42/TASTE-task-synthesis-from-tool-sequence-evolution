from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import os

from src.common.domain_utils import WORKSPACE_ROOT


def deep_diff(a: Any, b: Any, path: str = "") -> List[Dict[str, Any]]:
    """
    Recursively compare two JSON-like structures (dicts/lists/scalars).

    Returns a list of differences, each diff is a dict:
    - type: "added" | "removed" | "changed"
    - path: JSON-like path (e.g. "users[0].name")
    - old / new / value: depending on diff type
    """
    diffs: List[Dict[str, Any]] = []

    if a == b:
        return diffs

    if isinstance(a, dict) and isinstance(b, dict):
        all_keys = set(a.keys()) | set(b.keys())
        for key in all_keys:
            new_path = f"{path}.{key}" if path else str(key)

            if key not in b:
                diffs.append({
                    "type": "removed",
                    "path": new_path,
                    "value": a[key],
                })
            elif key not in a:
                diffs.append({
                    "type": "added",
                    "path": new_path,
                    "value": b[key],
                })
            else:
                diffs.extend(deep_diff(a[key], b[key], new_path))

    elif isinstance(a, list) and isinstance(b, list):
        max_len = max(len(a), len(b))
        for idx in range(max_len):
            new_path = f"{path}[{idx}]" if path else f"[{idx}]"
            if idx >= len(a):
                diffs.append({
                    "type": "added",
                    "path": new_path,
                    "value": b[idx],
                })
            elif idx >= len(b):
                diffs.append({
                    "type": "removed",
                    "path": new_path,
                    "value": a[idx],
                })
            else:
                diffs.extend(deep_diff(a[idx], b[idx], new_path))

    else:
        diffs.append({
            "type": "changed",
            "path": path or "$",
            "old": a,
            "new": b,
        })

    return diffs


def safe_model_dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            return obj
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def safe_json_dumps(payload: Any, indent: int = 2) -> str:
    try:
        return json.dumps(payload, indent=indent, default=str)
    except TypeError:
        return str(payload)


def normalize_action_payload(action: Any) -> Any:
    if action is None:
        return None

    action = safe_model_dump(action)

    if isinstance(action, str):
        try:
            action = json.loads(action)
        except Exception:
            return action

    if isinstance(action, dict):
        if "function" in action and isinstance(action["function"], dict):
            action = {
                "name": action["function"].get("name"),
                "arguments": action["function"].get("arguments"),
            }

        name = action.get("name") or action.get("tool_name") or action.get("action_name")
        arguments = action.get("arguments") or action.get("args") or action.get("parameters")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                pass

        if name is not None or arguments is not None:
            return {"name": name, "arguments": arguments}

    return action


def extract_agent_actions(conversation_messages: List[Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for msg in conversation_messages:
        if not isinstance(msg, dict):
            continue

        tool_calls = msg.get("tool_calls") or []
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]

        for tc in tool_calls:
            normalized = normalize_action_payload(tc)
            if isinstance(normalized, dict) and ("name" in normalized or "arguments" in normalized):
                actions.append(normalized)

        function_call = msg.get("function_call")
        if function_call:
            normalized = normalize_action_payload({"function": function_call})
            if isinstance(normalized, dict):
                actions.append(normalized)

    return actions


def extract_actual_action_from_check(action_check: Any) -> Optional[Any]:
    if action_check is None:
        return None

    candidate_keys = [
        "actual_action",
        "agent_action",
        "predicted_action",
        "model_action",
        "action_taken",
        "action_call",
        "tool_call",
        "assistant_action",
        "action_pred",
    ]

    raw = safe_model_dump(action_check)
    if isinstance(raw, dict):
        for key in candidate_keys:
            if key in raw:
                return normalize_action_payload(raw.get(key))

    for key in candidate_keys:
        if hasattr(action_check, key):
            return normalize_action_payload(getattr(action_check, key))

    return None


def raw_action_check_has_extra(raw_action_check: Any) -> bool:
    if raw_action_check is None:
        return False
    if not isinstance(raw_action_check, dict):
        return True

    baseline_keys = {"action", "action_match", "action_reward"}
    if any(key not in baseline_keys for key in raw_action_check.keys()):
        return True

    return False


def log_template_error(
    task_dict: Dict[str, Any],
    db_entities: Dict[str, Any],
    template_errors: List[str],
    domain: str = "airline",
) -> str:
    """Log template variable validation errors with full task and DB context."""
    log_dir = os.path.join(
        WORKSPACE_ROOT,
        "logs",
        domain,
        "template_validation_errors",
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = task_dict.get("id", "unknown")
    log_file = os.path.join(log_dir, f"template_error_{task_id}_{timestamp}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("TEMPLATE VARIABLE VALIDATION ERROR LOG\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task ID: {task_id}\n\n")

        f.write("TEMPLATE ERRORS\n")
        f.write("-" * 100 + "\n")
        for err in template_errors:
            f.write(f"  - {err}\n")
        f.write("\n")

        f.write("ACTION ARGUMENTS\n")
        f.write("-" * 100 + "\n")
        actions = task_dict.get("evaluation_criteria", {}).get("actions", [])
        for idx, action in enumerate(actions):
            f.write(f"  [{idx}] {action.get('name', 'unknown')}\n")
            f.write(f"      Arguments: {safe_json_dumps(action.get('arguments', {}), indent=10)}\n")
        f.write("\n")

        f.write("TASK DICT\n")
        f.write("-" * 100 + "\n")
        f.write(safe_json_dumps(task_dict))
        f.write("\n\n")

        f.write("DB ENTITIES\n")
        f.write("-" * 100 + "\n")
        f.write(safe_json_dumps(db_entities))
        f.write("\n\n")

        f.write("=" * 100 + "\n")

    print(f"  📝 Template error logged to: {log_file}")
    return log_file


def log_db_preflight_error(
    task_dict: Dict[str, Any],
    db_entities: Dict[str, Any],
    preflight_errors: List[str],
    domain: str = "airline",
) -> str:
    """Log DB preflight check errors with full task and DB context."""
    log_dir = os.path.join(
        WORKSPACE_ROOT,
        "logs",
        domain,
        "db_preflight_errors",
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = task_dict.get("id", "unknown")
    log_file = os.path.join(log_dir, f"preflight_error_{task_id}_{timestamp}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("DB PREFLIGHT CHECK ERROR LOG\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task ID: {task_id}\n\n")

        f.write("PREFLIGHT ERRORS\n")
        f.write("-" * 100 + "\n")
        for err in preflight_errors:
            f.write(f"  - {err}\n")
        f.write("\n")

        f.write("ACTION ARGUMENTS\n")
        f.write("-" * 100 + "\n")
        actions = task_dict.get("evaluation_criteria", {}).get("actions", [])
        for idx, action in enumerate(actions):
            f.write(f"  [{idx}] {action.get('name', 'unknown')}\n")
            f.write(f"      Arguments: {safe_json_dumps(action.get('arguments', {}), indent=10)}\n")
        f.write("\n")

        f.write("DB ENTITIES\n")
        f.write("-" * 100 + "\n")
        f.write(safe_json_dumps(db_entities))
        f.write("\n\n")

        f.write("=" * 100 + "\n")

    print(f"  📝 DB preflight error logged to: {log_file}")
    return log_file


def log_db_schema_error(
    task_dict: Dict[str, Any],
    db_entities: Dict[str, Any],
    schema_errors: List[str],
    domain: str = "airline",
) -> str:
    """Log DB schema validation errors with full DB context."""
    log_dir = os.path.join(
        WORKSPACE_ROOT,
        "logs",
        domain,
        "db_schema_errors",
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = task_dict.get("id", "unknown")
    log_file = os.path.join(log_dir, f"schema_error_{task_id}_{timestamp}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("DB SCHEMA VALIDATION ERROR LOG\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task ID: {task_id}\n\n")

        f.write("SCHEMA ERRORS\n")
        f.write("-" * 100 + "\n")
        for err in schema_errors:
            f.write(f"  - {err}\n")
        f.write("\n")

        f.write("DB ENTITIES\n")
        f.write("-" * 100 + "\n")
        f.write(safe_json_dumps(db_entities))
        f.write("\n\n")

        f.write("ACTION ARGUMENTS (for context)\n")
        f.write("-" * 100 + "\n")
        actions = task_dict.get("evaluation_criteria", {}).get("actions", [])
        for idx, action in enumerate(actions):
            f.write(f"  [{idx}] {action.get('name', 'unknown')}\n")
            f.write(f"      Arguments: {safe_json_dumps(action.get('arguments', {}), indent=10)}\n")
        f.write("\n")

        f.write("=" * 100 + "\n")

    print(f"  📝 DB schema error logged to: {log_file}")
    return log_file


def log_coherence_error(
    task_dict: Dict[str, Any],
    db_entities: Dict[str, Any],
    error: str,
    review_response: Optional[str] = None,
    domain: str = "airline",
) -> str:
    """Log coherence review errors with full task, DB, and LLM response context."""
    log_dir = os.path.join(
        WORKSPACE_ROOT,
        "logs",
        domain,
        "coherence_errors",
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = task_dict.get("id", "unknown")
    log_file = os.path.join(log_dir, f"coherence_error_{task_id}_{timestamp}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("COHERENCE REVIEW ERROR LOG\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task ID: {task_id}\n\n")

        f.write("ERROR\n")
        f.write("-" * 100 + "\n")
        f.write(f"{error}\n\n")

        if review_response:
            f.write("LLM REVIEW RESPONSE\n")
            f.write("-" * 100 + "\n")
            f.write(review_response)
            f.write("\n\n")

        f.write("TASK DICT\n")
        f.write("-" * 100 + "\n")
        f.write(safe_json_dumps(task_dict))
        f.write("\n\n")

        f.write("DB ENTITIES\n")
        f.write("-" * 100 + "\n")
        f.write(safe_json_dumps(db_entities))
        f.write("\n\n")

        f.write("=" * 100 + "\n")

    print(f"  📝 Coherence error logged to: {log_file}")
    return log_file


def log_gt_agent_error(
    task_dict: Dict[str, Any],
    simulation: Any,
    db_reward: Optional[float],
    action_rewards: List[float],
    error: Optional[str],
    reward_info: Any,
    gt_llm: str,
    domain: str = "airline",
) -> str:
    log_dir = os.path.join(
        WORKSPACE_ROOT,
        "logs",
        domain,
        "gt_agent_validation_errors",
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = task_dict.get("id", "unknown")
    log_file = os.path.join(log_dir, f"{timestamp}_gt_error_{task_id}.txt")

    conversation_messages: List[Any] = []
    messages = getattr(simulation, "messages", None)
    conversation = getattr(simulation, "conversation", None)

    if messages:
        for msg in messages:
            if hasattr(msg, "model_dump"):
                conversation_messages.append(msg.model_dump())
            elif hasattr(msg, "__dict__"):
                conversation_messages.append(msg.__dict__)
            elif isinstance(msg, dict):
                conversation_messages.append(msg)
            else:
                conversation_messages.append(str(msg))
    elif conversation:
        for msg in conversation:
            if hasattr(msg, "model_dump"):
                conversation_messages.append(msg.model_dump())
            elif hasattr(msg, "__dict__"):
                conversation_messages.append(msg.__dict__)
            elif isinstance(msg, dict):
                conversation_messages.append(msg)
            else:
                conversation_messages.append(str(msg))
    elif hasattr(simulation, "model_dump"):
        sim_dict = simulation.model_dump()
        conversation_messages = sim_dict.get("conversation", sim_dict.get("messages", []))

    eval_criteria = task_dict.get("evaluation_criteria", {})
    expected_actions = eval_criteria.get("actions", [])
    agent_actions = extract_agent_actions(conversation_messages)

    action_check_details = []
    mismatch_details = []
    if reward_info.action_checks is not None:
        for idx, ac in enumerate(reward_info.action_checks):
            action_reward = getattr(ac, "action_reward", None)
            action_match = getattr(ac, "action_match", None)
            expected_action = expected_actions[idx] if idx < len(expected_actions) else getattr(ac, "action", None)
            expected_action_norm = normalize_action_payload(expected_action)
            actual_action = extract_actual_action_from_check(ac)
            if actual_action is None and idx < len(agent_actions):
                actual_action = agent_actions[idx]
            actual_action_norm = normalize_action_payload(actual_action)
            diffs = []
            if expected_action_norm is not None and actual_action_norm is not None:
                diffs = deep_diff(expected_action_norm, actual_action_norm)

            action_check_details.append({
                "index": idx,
                "action_reward": action_reward,
                "action_match": action_match,
                "action": getattr(ac, "action", None),
            })

            if (action_reward is not None and action_reward < 1.0) or action_match is False:
                mismatch_details.append({
                    "index": idx,
                    "action_reward": action_reward,
                    "action_match": action_match,
                    "expected_action": expected_action_norm,
                    "actual_action": actual_action_norm,
                    "diffs": diffs,
                    "raw_action_check": safe_model_dump(ac),
                })

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("GT AGENT VALIDATION ERROR LOG\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task ID: {task_id}\n")
        f.write(f"GT LLM: {gt_llm}\n")
        f.write("\n" + "-" * 100 + "\n\n")

        f.write("TASK SUMMARY\n")
        f.write("-" * 100 + "\n")
        description = task_dict.get("description", {})
        f.write(f"Purpose: {description.get('purpose', 'N/A')}\n")
        user_scenario = task_dict.get("user_scenario", {})
        instructions = user_scenario.get("instructions", {})
        f.write(f"Reason for Call: {instructions.get('reason_for_call', 'N/A')}\n")
        known_info = instructions.get("known_info", "N/A")
        if len(str(known_info)) > 200:
            f.write(f"Known Info: {str(known_info)[:200]}...\n")
        else:
            f.write(f"Known Info: {known_info}\n")

        f.write(f"\nExpected Actions ({len(expected_actions)}):\n")
        for idx, action in enumerate(expected_actions):
            action_name = action.get("name", "unknown")
            f.write(f"  {idx}. {action_name}\n")

        f.write("\n" + "-" * 100 + "\n\n")

        f.write("EVALUATION CRITERIA\n")
        f.write("-" * 100 + "\n")

        database_conditions = eval_criteria.get("database", {})
        if database_conditions:
            f.write("Database Conditions:\n")
            f.write(json.dumps(database_conditions, indent=2))
            f.write("\n\n")
        else:
            f.write("Database Conditions: None\n\n")

        if expected_actions:
            f.write("Expected Action Sequence:\n")
            for idx, action in enumerate(expected_actions):
                f.write(f"  [{idx}] {action.get('name', 'unknown')}\n")
                arguments = action.get("arguments", {})
                if arguments:
                    f.write(f"      Arguments: {json.dumps(arguments, indent=10)}\n")
                else:
                    f.write("      Arguments: None\n")
        else:
            f.write("Expected Action Sequence: None\n")

        if agent_actions:
            f.write("\nAgent Action Sequence (from tool calls):\n")
            for idx, action in enumerate(agent_actions):
                if isinstance(action, dict):
                    f.write(f"  [{idx}] {action.get('name', 'unknown')}\n")
                    arguments = action.get("arguments")
                    if arguments is not None:
                        f.write(f"      Arguments: {safe_json_dumps(arguments, indent=10)}\n")
                    else:
                        f.write("      Arguments: None\n")
                else:
                    f.write(f"  [{idx}] {safe_json_dumps(action, indent=2)}\n")
        else:
            f.write("\nAgent Action Sequence (from tool calls): None\n")

        f.write("\n" + "-" * 100 + "\n\n")

        f.write("ERROR DETAILS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Error Message: {error}\n\n")
        f.write(f"DB Reward: {db_reward} (expected: 1.0)\n")
        f.write(f"DB Match: {getattr(reward_info.db_check, 'db_match', None) if reward_info.db_check else None}\n")
        f.write(f"\nAction Rewards: {action_rewards}\n")
        f.write(f"All Actions Passed: {all(r == 1.0 for r in action_rewards) if action_rewards else False}\n")

        if action_check_details:
            f.write("\nAction Check Details:\n")
            for ac_detail in action_check_details:
                f.write(f"  Action {ac_detail['index']}:\n")
                f.write(f"    Reward: {ac_detail['action_reward']}\n")
                f.write(f"    Match: {ac_detail['action_match']}\n")
                if ac_detail.get("action"):
                    action = ac_detail["action"]
                    if isinstance(action, dict):
                        f.write(f"    Name: {action.get('name', 'N/A')}\n")
                        f.write(f"    Arguments: {json.dumps(action.get('arguments', {}), indent=6)}\n")

        if mismatch_details:
            f.write("\nACTION MISMATCH DETAILS\n")
            f.write("-" * 100 + "\n")
            f.write(f"Agent Tool Calls Found: {len(agent_actions)}\n")
            for detail in mismatch_details:
                f.write(f"  Action {detail['index']}:\n")
                f.write(f"    Reward: {detail['action_reward']}\n")
                f.write(f"    Match: {detail['action_match']}\n")
                if detail.get("expected_action") is not None:
                    f.write(f"    Expected Action: {safe_json_dumps(detail['expected_action'], indent=6)}\n")
                else:
                    f.write("    Expected Action: Not available\n")
                if detail.get("actual_action") is not None:
                    f.write(f"    Actual Action: {safe_json_dumps(detail['actual_action'], indent=6)}\n")
                else:
                    f.write("    Actual Action: Not available\n")
                if detail.get("diffs"):
                    f.write("    Diff:\n")
                    for diff in detail["diffs"]:
                        if diff["type"] == "changed":
                            f.write(f"      - {diff['path']}: {diff.get('old')} -> {diff.get('new')}\n")
                        elif diff["type"] in ("added", "removed"):
                            f.write(f"      - {diff['type']} {diff['path']}: {diff.get('value')}\n")
                else:
                    f.write("    Diff: None\n")
                raw_action_check = detail.get("raw_action_check")
                if detail.get("actual_action") is None and raw_action_check_has_extra(raw_action_check):
                    f.write(f"    Raw Action Check: {safe_json_dumps(raw_action_check, indent=6)}\n")
                elif detail.get("actual_action") is None:
                    f.write("    Raw Action Check: omitted (no extra fields)\n")
                f.write("\n")

        f.write("\n" + "-" * 100 + "\n\n")

        f.write("CONVERSATION MESSAGES\n")
        f.write("-" * 100 + "\n")
        if conversation_messages:
            f.write(f"Total Messages: {len(conversation_messages)}\n\n")
            for idx, msg in enumerate(conversation_messages):
                f.write(f"[Message {idx + 1}]\n")
                if isinstance(msg, dict):
                    role = msg.get("role", msg.get("type", "unknown"))
                    f.write(f"Role: {role}\n")

                    content = msg.get("content", msg.get("text", ""))
                    if content:
                        if len(str(content)) > 500:
                            f.write(f"Content: {str(content)[:500]}...\n")
                        else:
                            f.write(f"Content: {content}\n")

                    tool_calls = msg.get("tool_calls", msg.get("tool_calls", []))
                    if tool_calls:
                        f.write(f"Tool Calls ({len(tool_calls)}):\n")
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                f.write(f"  - {tc.get('name', 'unknown')}: {json.dumps(tc.get('arguments', {}), indent=4)}\n")
                            else:
                                f.write(f"  - {tc}\n")

                    if "id" in msg:
                        f.write(f"ID: {msg['id']}\n")
                else:
                    f.write(f"{msg}\n")
                f.write("\n")
        else:
            f.write("No conversation messages available.\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF ERROR LOG\n")
        f.write("=" * 100 + "\n")

    print(f"  📝 GT agent error logged to: {log_file}")
    return log_file


def log_solver_error(
    action_arguments: List[Dict[str, Any]],
    error: str,
    task_dict: Optional[Dict[str, Any]] = None,
    domain: str = "airline",
) -> str:
    log_dir = os.path.join(
        WORKSPACE_ROOT,
        "logs",
        domain,
        "solver_validation_errors",
    )
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = task_dict.get("id", "unknown") if task_dict else "unknown"
    log_file = os.path.join(log_dir, f"solver_error_{task_id}_{timestamp}.txt")

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("SOLVER VALIDATION ERROR LOG\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Task ID: {task_id}\n")
        f.write("\n" + "-" * 100 + "\n\n")

        if task_dict:
            f.write("TASK SUMMARY\n")
            f.write("-" * 100 + "\n")
            description = task_dict.get("description", {})
            f.write(f"Purpose: {description.get('purpose', 'N/A')}\n")
            user_scenario = task_dict.get("user_scenario", {})
            instructions = user_scenario.get("instructions", {})
            f.write(f"Reason for Call: {instructions.get('reason_for_call', 'N/A')}\n")
            f.write("\n" + "-" * 100 + "\n\n")

        f.write("ERROR DETAILS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Error Message: {error}\n")
        f.write("\n" + "-" * 100 + "\n\n")

        if task_dict:
            eval_criteria = task_dict.get("evaluation_criteria", {})
            if eval_criteria:
                f.write("EVALUATION CRITERIA\n")
                f.write("-" * 100 + "\n")

                database_conditions = eval_criteria.get("database", {})
                if database_conditions:
                    f.write("Database Conditions:\n")
                    f.write(json.dumps(database_conditions, indent=2))
                    f.write("\n\n")
                else:
                    f.write("Database Conditions: None\n\n")

                expected_actions = eval_criteria.get("actions", [])
                if expected_actions:
                    f.write("Expected Action Sequence:\n")
                    for idx, action in enumerate(expected_actions):
                        f.write(f"  [{idx}] {action.get('name', 'unknown')}\n")
                        arguments = action.get("arguments", {})
                        if arguments:
                            f.write(f"      Arguments: {json.dumps(arguments, indent=10)}\n")
                        else:
                            f.write("      Arguments: None\n")
                else:
                    f.write("Expected Action Sequence: None\n")

                f.write("\n" + "-" * 100 + "\n\n")

        f.write("ACTION ARGUMENTS\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total Actions: {len(action_arguments)}\n\n")
        for idx, action in enumerate(action_arguments):
            f.write(f"[Action {idx}]\n")
            f.write(f"  Name: {action.get('name', 'unknown')}\n")
            f.write(f"  Arguments: {json.dumps(action.get('arguments', {}), indent=4)}\n")
            f.write("\n")

        f.write("\n" + "-" * 100 + "\n\n")

        if task_dict:
            initial_state = task_dict.get("initial_state", {})
            if initial_state:
                f.write("INITIAL DB SUMMARY\n")
                f.write("-" * 100 + "\n")
                for key, value in initial_state.items():
                    if isinstance(value, list):
                        f.write(f"  {key}: {len(value)} entries\n")
                    else:
                        f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF ERROR LOG\n")
        f.write("=" * 100 + "\n")

    print(f"  📝 Solver error logged to: {log_file}")
    return log_file
