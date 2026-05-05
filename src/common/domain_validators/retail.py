"""Retail-domain-specific validation logic.

Lean implementation: basic schema checks + solver + GT agent.
No deep preflight validation. Intended to evolve based on
observed LLM failure patterns.
"""

from typing import Any, Dict, List

from src.common.domain_validators.base import BaseDomainValidator


# ---------------------------------------------------------------------------
# Structural pre-filter for pool generation
# ---------------------------------------------------------------------------

_AUTH_ACTIONS = {"find_user_id_by_email", "find_user_id_by_name_zip"}

_ORDER_WRITE_ACTIONS = {
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}

_NEEDS_PRODUCT_LOOKUP = {
    "exchange_delivered_order_items",
    "modify_pending_order_items",
}

_PRODUCT_LOOKUP_ACTIONS = {"get_product_details", "list_all_product_types"}

_RETURN_EXCHANGE_ACTIONS = {
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}


def retail_structural_filter(sequence: List[str]) -> bool:
    """Fast structural pre-filter for retail action sequences.

    Returns True if the sequence passes basic structural checks,
    False if it is obviously invalid. Designed to run in microseconds
    so the pool generator can discard broken sequences cheaply before
    LLM validation.
    """
    if not sequence:
        return True

    n = len(sequence)

    # --- Rule 1: auth must be at the start; consecutive retries allowed ---
    auth_indices = [i for i, a in enumerate(sequence) if a in _AUTH_ACTIONS]
    if auth_indices:
        if auth_indices[0] != 0:
            return False
        # All auth actions must be consecutive at the start (retries)
        for idx in range(1, len(auth_indices)):
            if auth_indices[idx] != auth_indices[idx - 1] + 1:
                return False
    has_auth = len(auth_indices) >= 1

    # --- Rule 2: transfer_to_human_agents must be last ---
    for i, a in enumerate(sequence):
        if a == "transfer_to_human_agents" and i != n - 1:
            return False

    # --- Rules 3 & 4 apply only when auth is present ---
    if has_auth:
        seen_order_lookup = False
        seen_product_lookup = False

        for a in sequence:
            if a == "get_order_details":
                seen_order_lookup = True
            elif a in _PRODUCT_LOOKUP_ACTIONS:
                seen_product_lookup = True
            elif a in _ORDER_WRITE_ACTIONS:
                # Rule 3: order-write needs get_order_details before it
                if not seen_order_lookup:
                    return False
                # Rule 4: exchange/modify_items needs product lookup before it
                if a in _NEEDS_PRODUCT_LOOKUP and not seen_product_lookup:
                    return False

    # --- Rule 5: no return + exchange without intervening get_order_details ---
    # Only when auth is present; sub-task sequences get benefit of the doubt.
    if has_auth:
        last_return_or_exchange = None
        for a in sequence:
            if a == "get_order_details":
                last_return_or_exchange = None
            elif a in _RETURN_EXCHANGE_ACTIONS:
                if last_return_or_exchange is not None and last_return_or_exchange != a:
                    return False
                last_return_or_exchange = a

    return True


class RetailValidator(BaseDomainValidator):
    """Lean retail validator: basic schema checks only."""

    def check_preflight(
        self, action_arguments: List[Dict[str, Any]], db_entities: Dict[str, Any]
    ) -> List[str]:
        """Basic preflight: verify referenced entity IDs exist in DB entities."""
        errors = []

        users = db_entities.get("users", {})
        orders = db_entities.get("orders", {})
        products = db_entities.get("products", {})

        # Collect all item_ids across all products
        all_item_ids = set()
        for product in products.values():
            if isinstance(product, dict):
                for variant_id in product.get("variants", {}).keys():
                    all_item_ids.add(variant_id)

        for i, action in enumerate(action_arguments):
            name = action.get("name", "")
            args = action.get("arguments", {})

            if "user_id" in args:
                uid = args["user_id"]
                if uid and uid not in users:
                    errors.append(f"Action {i} ({name}): user_id '{uid}' not found in DB entities")

            if "order_id" in args:
                oid = args["order_id"]
                if oid and oid not in orders:
                    errors.append(f"Action {i} ({name}): order_id '{oid}' not found in DB entities")

            for key in ("item_ids", "new_item_ids"):
                if key in args and isinstance(args[key], list):
                    for item_id in args[key]:
                        if item_id and all_item_ids and item_id not in all_item_ids:
                            errors.append(f"Action {i} ({name}): {key} contains '{item_id}' not found in any product variant")

        return errors

    def validate_db_schema(self, db_entities: Dict[str, Any]) -> List[str]:
        """Basic schema: check required keys and types."""
        errors = []

        for key in ("users", "orders", "products"):
            if key not in db_entities:
                errors.append(f"Missing required DB entity type: '{key}'")
            elif not isinstance(db_entities[key], dict):
                errors.append(f"DB entity '{key}' must be a dict, got {type(db_entities[key]).__name__}")

        orders = db_entities.get("orders", {})
        for order_id, order in orders.items():
            if not isinstance(order, dict):
                errors.append(f"Order '{order_id}' must be a dict")
                continue
            if "status" not in order:
                errors.append(f"Order '{order_id}' missing 'status' field")
            if "items" not in order:
                errors.append(f"Order '{order_id}' missing 'items' field")
            if "user_id" not in order:
                errors.append(f"Order '{order_id}' missing 'user_id' field")

        return errors
