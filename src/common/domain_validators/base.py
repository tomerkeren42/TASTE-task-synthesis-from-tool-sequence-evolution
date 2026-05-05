"""Base class for domain-specific validation logic."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BaseDomainValidator(ABC):
    """Domain-specific validation logic called by TaskValidator."""

    def __init__(self, domain_config):
        self.domain_config = domain_config

    @abstractmethod
    def check_preflight(
        self, action_arguments: List[Dict[str, Any]], db_entities: Dict[str, Any]
    ) -> List[str]:
        """Domain-specific preflight checks. Return list of error strings."""

    @abstractmethod
    def validate_db_schema(self, db_entities: Dict[str, Any]) -> List[str]:
        """Validate DB entities against domain schema. Return list of error strings."""

    def build_environment(self, db_state: Dict[str, Any]) -> Tuple[Any, Any]:
        """Build (db_model, environment) from raw db_state dict."""
        db_class = self.domain_config.get_db_class()
        db = db_class.model_validate(db_state)
        env = self.domain_config.get_environment(db=db)
        return db, env

    def get_empty_db_dict(self) -> Dict[str, Any]:
        """Return the default empty DB state as a dict."""
        return self.domain_config.get_db().model_dump()
