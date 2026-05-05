"""Telecom-domain-specific validation logic.

Validates telecom tasks against TelecomDB (backend) and TelecomUserDB
(device/surroundings) state. Supports dual-DB environment construction.
"""

from typing import Any, Dict, List, Tuple

from src.common.domain_validators.base import BaseDomainValidator


# ---------------------------------------------------------------------------
# Action sets for structural pre-filter
# ---------------------------------------------------------------------------

_LOOKUP_ACTIONS = {
    "get_customer_by_phone",
    "get_customer_by_id",
    "get_customer_by_name",
}

_AGENT_BACKEND_ACTIONS = {
    "suspend_line", "resume_line",
    "enable_roaming", "disable_roaming",
    "refuel_data", "send_payment_request",
    "get_details_by_id", "get_bills_for_customer", "get_data_usage",
}

# Agent actions that have a corresponding user-side follow-up
_AGENT_USER_PAIRS = {
    "enable_roaming": "toggle_roaming",
    "disable_roaming": "toggle_roaming",
    "send_payment_request": "make_payment",
}


def telecom_structural_filter(sequence: List[str]) -> bool:
    """Fast structural pre-filter for telecom action sequences.

    Returns True if the sequence passes basic structural checks,
    False if it is obviously invalid.
    """
    if not sequence:
        return True

    n = len(sequence)

    # Rule 1: transfer_to_human_agents must be last if present
    for i, a in enumerate(sequence):
        if a == "transfer_to_human_agents" and i != n - 1:
            return False

    # Rule 2: lookup actions should appear in the first half
    lookup_indices = [i for i, a in enumerate(sequence) if a in _LOOKUP_ACTIONS]
    if lookup_indices:
        if min(lookup_indices) > n // 2:
            return False

    # Rule 3: agent backend action should come before its paired user action
    for agent_action, user_action in _AGENT_USER_PAIRS.items():
        agent_idx = None
        for i, a in enumerate(sequence):
            if a == agent_action:
                agent_idx = i
            elif a == user_action and agent_idx is None:
                remaining = sequence[i + 1:]
                if agent_action in remaining:
                    return False

    return True


class TelecomValidator(BaseDomainValidator):
    """Telecom validator: entity existence, referential integrity, dual-DB."""

    # Maps wrong LLM-generated field names → correct Pydantic field names.
    # Applied automatically before validation/environment construction.
    _DEVICE_FIELD_ALIASES = {
        "sim_status": "sim_card_status",
        "sim_missing": "sim_card_missing",
        "mobile_data_enabled": "data_enabled",
        "mobile_data": "data_enabled",
        "data_roaming_enabled": "roaming_enabled",
        "data_roaming": "roaming_enabled",
        "apn_settings": "active_apn_settings",
        "apn": "active_apn_settings",
        "wifi_status": "wifi_connected",
        "wifi_connection_status": "wifi_connected",
        "vpn_status": "vpn_connected",
        "vpn_connection_status": "vpn_connected",
        "installed_apps": "app_statuses",
        "apps": "app_statuses",
        "network_type": "network_technology_connected",
        "mobile_network_type": "network_technology_connected",
        "preferred_network_mode": "network_mode_preference",
        "device_id": None,  # not a device field, drop silently
    }

    _LINE_FIELD_ALIASES = {
        "roaming_status": "roaming_enabled",
        "is_roaming_enabled": "roaming_enabled",
    }

    @staticmethod
    def _fix_field_names(db_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-correct common LLM field name hallucinations.

        Mutates and returns *db_entities* with wrong field names replaced
        by the correct Pydantic field names. Silently drops fields that
        map to None.
        """
        # Fix device fields
        device = db_entities.get("device")
        if isinstance(device, dict):
            for wrong, correct in TelecomValidator._DEVICE_FIELD_ALIASES.items():
                if wrong in device and correct not in device:
                    if correct is not None:
                        device[correct] = device.pop(wrong)
                    else:
                        device.pop(wrong)
                elif wrong in device and correct in device:
                    device.pop(wrong)  # drop duplicate

            # Convert list-format app_statuses to dict-of-dicts
            apps = device.get("app_statuses")
            if isinstance(apps, list):
                device["app_statuses"] = {
                    item.get("app_name", f"app_{i}"): item
                    for i, item in enumerate(apps)
                    if isinstance(item, dict)
                }
                apps = device["app_statuses"]

            # Strip invalid permission keys from app_statuses
            _valid_perms = {"sms", "storage", "phone", "network"}
            if isinstance(apps, dict):
                for app_data in apps.values():
                    if isinstance(app_data, dict) and "permissions" in app_data:
                        perms = app_data["permissions"]
                        if isinstance(perms, dict):
                            for k in list(perms.keys()):
                                if k not in _valid_perms:
                                    perms.pop(k)

        # Fix surroundings fields (LLM sometimes puts device fields here)
        surroundings = db_entities.get("surroundings")
        if isinstance(surroundings, dict):
            for wrong in list(surroundings.keys()):
                if wrong in TelecomValidator._DEVICE_FIELD_ALIASES:
                    surroundings.pop(wrong)

        # Fix line fields
        def _fix_line(line: dict) -> None:
            for wrong, correct in TelecomValidator._LINE_FIELD_ALIASES.items():
                if wrong in line and correct not in line:
                    line[correct] = line.pop(wrong)
                elif wrong in line:
                    line.pop(wrong)
            # Normalise roaming_enabled string values to bool
            re = line.get("roaming_enabled")
            if isinstance(re, str):
                line["roaming_enabled"] = re.lower() in ("true", "enabled", "on", "yes")

        lines = db_entities.get("lines")
        if isinstance(lines, dict):
            for lid, line in lines.items():
                if isinstance(line, dict):
                    _fix_line(line)
        elif isinstance(lines, list):
            for line in lines:
                if isinstance(line, dict):
                    _fix_line(line)

        return db_entities

    # Maps entity_type -> id_field_name for normalisation.
    _ENTITY_ID_FIELDS = {
        "customers": "customer_id",
        "lines": "line_id",
        "bills": "bill_id",
        "devices": "device_id",
        "plans": "plan_id",
    }

    @staticmethod
    def _to_dict_of_dicts(entity_type: str, entities: Any) -> Dict[str, Any]:
        """Normalise entities to dict-of-dicts keyed by entity ID.

        Accepts either:
        - dict-of-dicts (LLM output): ``{"C1001": {...}, ...}``
        - list-of-dicts (model_dump output): ``[{"customer_id": "C1001", ...}, ...]``

        Returns dict-of-dicts in both cases.
        """
        if isinstance(entities, dict):
            return entities
        if isinstance(entities, list):
            id_field = TelecomValidator._ENTITY_ID_FIELDS.get(entity_type)
            if not id_field:
                return {}
            return {item[id_field]: item for item in entities if isinstance(item, dict) and id_field in item}
        return {}

    def check_preflight(
        self, action_arguments: List[Dict[str, Any]], db_entities: Dict[str, Any]
    ) -> List[str]:
        """Validate referenced entity IDs exist and relationships are correct."""
        errors = []

        customers = self._to_dict_of_dicts("customers", db_entities.get("customers", {}))
        lines = self._to_dict_of_dicts("lines", db_entities.get("lines", {}))
        bills = self._to_dict_of_dicts("bills", db_entities.get("bills", {}))
        devices = self._to_dict_of_dicts("devices", db_entities.get("devices", {}))
        plans = self._to_dict_of_dicts("plans", db_entities.get("plans", {}))

        customer_ids = set(customers.keys())
        line_ids = set(lines.keys())
        bill_ids = set(bills.keys())
        device_ids = set(devices.keys())
        plan_ids = set(plans.keys())

        customer_line_ids = {}
        for cid, cust in customers.items():
            if isinstance(cust, dict):
                customer_line_ids[cid] = set(cust.get("line_ids", []))

        customer_bill_ids = {}
        for cid, cust in customers.items():
            if isinstance(cust, dict):
                customer_bill_ids[cid] = set(cust.get("bill_ids", []))

        for i, action in enumerate(action_arguments):
            name = action.get("name", "")
            args = action.get("arguments", {})

            if "customer_id" in args:
                cid = args["customer_id"]
                if cid and cid not in customer_ids:
                    errors.append(
                        f"Action {i} ({name}): customer_id '{cid}' not found in DB"
                    )

            if "line_id" in args:
                lid = args["line_id"]
                if lid and lid not in line_ids:
                    errors.append(
                        f"Action {i} ({name}): line_id '{lid}' not found in DB"
                    )
                cid = args.get("customer_id")
                if lid and cid and cid in customer_line_ids:
                    if lid not in customer_line_ids[cid]:
                        errors.append(
                            f"Action {i} ({name}): line_id '{lid}' not owned by customer '{cid}'"
                        )

            if "bill_id" in args:
                bid = args["bill_id"]
                if bid and bid not in bill_ids:
                    errors.append(
                        f"Action {i} ({name}): bill_id '{bid}' not found in DB"
                    )
                cid = args.get("customer_id")
                if bid and cid and cid in customer_bill_ids:
                    if bid not in customer_bill_ids[cid]:
                        errors.append(
                            f"Action {i} ({name}): bill_id '{bid}' not owned by customer '{cid}'"
                        )

            if name == "get_details_by_id" and "id" in args:
                eid = args["id"]
                if eid:
                    all_ids = customer_ids | line_ids | bill_ids | device_ids | plan_ids
                    if eid not in all_ids:
                        errors.append(
                            f"Action {i} ({name}): id '{eid}' not found in any DB entity"
                        )

        return errors

    def validate_db_schema(self, db_entities: Dict[str, Any]) -> List[str]:
        """Validate TelecomDB schema: required types, referential integrity."""
        self._fix_field_names(db_entities)
        errors = []

        for key in ("customers", "plans", "lines", "bills", "devices"):
            if key not in db_entities:
                errors.append(f"Missing required DB entity type: '{key}'")
            elif not isinstance(db_entities[key], (dict, list)):
                errors.append(
                    f"DB entity '{key}' must be a dict or list, got {type(db_entities[key]).__name__}"
                )

        customers = self._to_dict_of_dicts("customers", db_entities.get("customers", {}))
        lines = self._to_dict_of_dicts("lines", db_entities.get("lines", {}))
        plans = self._to_dict_of_dicts("plans", db_entities.get("plans", {}))

        for cid, cust in customers.items():
            if not isinstance(cust, dict):
                errors.append(f"Customer '{cid}' must be a dict")
                continue
            for field in ("full_name", "email", "phone_number"):
                if field not in cust:
                    errors.append(f"Customer '{cid}' missing '{field}' field")
            for lid in cust.get("line_ids", []):
                if lid not in lines:
                    errors.append(
                        f"Customer '{cid}' references line_id '{lid}' not in lines"
                    )

        for lid, line in lines.items():
            if not isinstance(line, dict):
                errors.append(f"Line '{lid}' must be a dict")
                continue
            if "plan_id" in line and line["plan_id"] not in plans:
                errors.append(
                    f"Line '{lid}' references plan_id '{line['plan_id']}' not in plans"
                )

        # Validate user DB fields if present
        if "surroundings" in db_entities:
            surroundings = db_entities["surroundings"]
            if isinstance(surroundings, dict):
                phone = surroundings.get("phone_number")
                if phone:
                    line_phones = set()
                    for lid, line in lines.items():
                        if isinstance(line, dict) and "phone_number" in line:
                            line_phones.add(line["phone_number"])
                    if phone not in line_phones:
                        errors.append(
                            f"Surroundings phone_number '{phone}' does not match any line"
                        )

        return errors

    # Entity types whose ID field is used as the dict key in LLM output.
    # Maps entity_type -> id_field_name.
    _ENTITY_ID_FIELDS = {
        "customers": "customer_id",
        "lines": "line_id",
        "bills": "bill_id",
        "devices": "device_id",
        "plans": "plan_id",
    }

    @staticmethod
    def _dict_to_list(entity_type: str, entities: Any) -> Any:
        """Convert dict-of-dicts to list-of-dicts for TelecomDB fields.

        The LLM generates entities as ``{"C1001": {...}, "C1002": {...}}``
        but TelecomDB expects ``[{...}, {...}]``.  Each entity dict gets
        its ID field injected from the dict key if not already present.

        If *entities* is already a list it is returned unchanged.
        """
        if isinstance(entities, list):
            return entities
        if not isinstance(entities, dict):
            return entities
        id_field = TelecomValidator._ENTITY_ID_FIELDS.get(entity_type)
        result = []
        for eid, edata in entities.items():
            if isinstance(edata, dict):
                if id_field and id_field not in edata:
                    edata = {id_field: eid, **edata}
                result.append(edata)
        return result

    def build_environment(self, db_state: Dict[str, Any]) -> Tuple[Any, Any]:
        """Build (db_model, environment) from raw db_state dict.

        For telecom, db_state may contain both TelecomDB fields and
        user DB fields ('device', 'surroundings'). Splits them apart,
        converts dict-of-dicts to lists (TelecomDB uses List fields),
        and passes both to get_environment.
        """
        self._fix_field_names(db_state)
        user_db_keys = set(self.domain_config.constants.get("user_db_entity_types", []))

        user_db_fields = {}
        main_db_fields = {}
        for key, value in db_state.items():
            if key in user_db_keys:
                user_db_fields[key] = value
            else:
                main_db_fields[key] = self._dict_to_list(key, value)

        db_class = self.domain_config.get_db_class()
        db = db_class.model_validate(main_db_fields)

        user_db = None
        if user_db_fields and self.domain_config.has_user_db():
            user_db_class = self.domain_config.get_user_db_class()
            user_db = user_db_class.model_validate(user_db_fields)

        env = self.domain_config.get_environment(db=db, user_db=user_db)
        return db, env
