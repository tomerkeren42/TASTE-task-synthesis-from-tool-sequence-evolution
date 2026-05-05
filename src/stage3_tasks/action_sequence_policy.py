"""Deterministic policy-completion rules for task action sequences.

When the medoid/seed sampler (or any LLM task generator) produces an action
sequence that omits a policy-mandated follow-up, downstream replay ends up in
a state that contradicts the scenario intent. For example: telecom policy
says "After you resume the line, the user will have to reboot their device
to get service" — without the reboot, the gold-env end-state keeps
``network_connection_status='no_service'`` even though the line is Active.

This module applies domain-specific rules that insert the missing
follow-ups into an action list. It is intentionally small and deterministic;
anything requiring scenario-context judgement belongs in the scenario
refinement / LLM pipeline.
"""
from typing import Any, Dict, List, Tuple


def _insert_user_action_after(
    actions: List[Dict[str, Any]], index: int, name: str, tag: str
) -> Dict[str, Any]:
    """Return a synthetic user action to insert right after ``index``.

    Preserves the ``action_id`` naming convention by generating a derived id
    from the anchor action's own action_id (or its positional index).
    """
    anchor_id = actions[index].get("action_id", index)
    return {
        "action_id": f"policy_{tag}_after_{anchor_id}",
        "requestor": "user",
        "name": name,
        "arguments": {},
        "info": None,
        "compare_args": None,
    }


def _telecom_ensure_reboot_after_resume(
    actions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Insert ``reboot_device`` (user) after every ``resume_line`` (assistant)
    that is not already followed by a user reboot.

    "Followed by" means: the first user action after the resume in the
    sequence. If that first user action is ``reboot_device``, no insertion
    is needed. Otherwise, insert the reboot immediately after the resume.
    """
    patched: List[Dict[str, Any]] = []
    patches: List[str] = []
    i = 0
    while i < len(actions):
        a = actions[i]
        patched.append(a)
        is_resume = (
            a.get("name") == "resume_line"
            and a.get("requestor") == "assistant"
        )
        if is_resume:
            # Peek forward for the NEXT user action
            next_user_is_reboot = False
            j = i + 1
            while j < len(actions):
                nxt = actions[j]
                if nxt.get("requestor") == "user":
                    next_user_is_reboot = nxt.get("name") == "reboot_device"
                    break
                j += 1
            if not next_user_is_reboot:
                patched.append(_insert_user_action_after(actions, i, "reboot_device", "reboot"))
                patches.append(
                    f"inserted reboot_device after resume_line at index {i}"
                )
        i += 1
    return patched, patches


_DOMAIN_RULES = {
    "telecom": [
        _telecom_ensure_reboot_after_resume,
    ],
}


def apply_policy_completion(
    actions: List[Dict[str, Any]], domain: str
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return ``(patched_actions, notes)``.

    ``notes`` describes any insertions made. Empty list if nothing changed.
    Domains with no registered rules return the input untouched.
    """
    rules = _DOMAIN_RULES.get(domain, [])
    out = actions
    all_notes: List[str] = []
    for rule in rules:
        out, notes = rule(out)
        all_notes.extend(notes)
    return out, all_notes
