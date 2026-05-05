import json
from typing import Dict, Any


class DatabaseStateManager:
    """
    Manages database state initialization, applying entities, and computing changes.

    Works generically across domains: merges LLM-generated entity dicts into the
    base database state by entity type and ID, without assuming any specific
    entity structure.
    """

    def __init__(self, initial_db_state: Dict[str, Any]):
        """
        Initialize the database state manager.

        Args:
            initial_db_state: The initial database state
        """
        self.initial_db_state = initial_db_state

    def apply_entities(self, db_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Apply LLM-generated entities to the base database state.

        Merges entities generically by iterating over top-level keys in db_entities.
        For dict-of-dicts (keyed by entity ID): merges by entity ID.
        For other values (flat dicts, scalars): deep-merges or overwrites directly.
        """
        initialized_db = json.loads(json.dumps(self.initial_db_state, default=str))

        for entity_type, entities in db_entities.items():
            if entity_type not in initialized_db:
                initialized_db[entity_type] = entities
                continue

            existing = initialized_db[entity_type]

            # Dict-of-dicts: merge by entity ID (original behavior)
            if isinstance(entities, dict) and isinstance(existing, dict):
                # Check if this looks like entity-ID-keyed (values are dicts)
                # vs a flat attribute dict (values are scalars/mixed)
                values_are_dicts = all(
                    isinstance(v, dict) for v in entities.values()
                ) if entities else False
                existing_values_are_dicts = all(
                    isinstance(v, dict) for v in existing.values()
                ) if existing else False

                if values_are_dicts and existing_values_are_dicts:
                    # Entity-ID-keyed: merge by ID
                    for entity_id, entity_data in entities.items():
                        if entity_id not in existing:
                            existing[entity_id] = entity_data
                        else:
                            self._merge_entity(existing[entity_id], entity_data)
                else:
                    # Flat attribute dict (e.g., device, surroundings): deep merge
                    self._merge_entity(existing, entities)
            else:
                # Scalar or type mismatch: overwrite
                initialized_db[entity_type] = entities

        return initialized_db

    @staticmethod
    def _merge_entity(existing: Dict[str, Any], new_data: Dict[str, Any]) -> None:
        """Merge new_data fields into existing entity dict in place."""
        for key, value in new_data.items():
            if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
                existing[key].update(value)
            elif key in existing and isinstance(existing[key], list) and isinstance(value, list):
                existing_set = set(str(v) for v in existing[key])
                for item in value:
                    if str(item) not in existing_set:
                        existing[key].append(item)
            else:
                existing[key] = value
