"""Central configuration object for domain-specific resources."""

import importlib
import json
import os
from typing import Any, Dict, Optional

from src.common.domain_utils import WORKSPACE_ROOT

_ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, "artifacts")
_DOMAINS_DIR = os.path.join(_ARTIFACTS_DIR, "domains")


class DomainConfig:
    """Bundles all domain-specific resources into one object.

    Created once per run and threaded through to all components
    that need domain-specific behavior.
    """

    def __init__(self, domain: str):
        self.domain = domain
        self.domain_dir = os.path.join(_DOMAINS_DIR, domain)

        if not os.path.isdir(self.domain_dir):
            raise ValueError(
                f"Domain directory not found: {self.domain_dir}. "
                f"Available domains: {self.list_domains()}"
            )

        # Prompts are shared across domains under ``artifacts/prompts/``,
        # organized by stage (``stage2/``, ``stage3/``) with optional
        # per-domain overrides in ``stage*/<domain>/``.
        self.prompts_dir = os.path.join(_ARTIFACTS_DIR, "prompts")
        self.tool_spec_path = os.path.join(self.domain_dir, "tool_spec.json")

        domain_json_path = os.path.join(self.domain_dir, "domain.json")
        if os.path.exists(domain_json_path):
            with open(domain_json_path) as f:
                self.constants: Dict[str, Any] = json.load(f)
        else:
            self.constants = {}

    @staticmethod
    def list_domains():
        """Return list of available domain names."""
        if not os.path.isdir(_DOMAINS_DIR):
            return []
        return [
            d for d in os.listdir(_DOMAINS_DIR)
            if os.path.isdir(os.path.join(_DOMAINS_DIR, d))
            and os.path.isfile(os.path.join(_DOMAINS_DIR, d, "domain.json"))
        ]

    def get_db_class(self):
        """Dynamically import and return the DB Pydantic model class."""
        module_name = self.constants.get("data_model_module")
        class_name = self.constants.get("db_class")
        if not module_name or not class_name:
            raise ValueError(
                f"domain.json for '{self.domain}' missing data_model_module or db_class"
            )
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def get_db(self):
        """Return an empty DB instance for this domain."""
        module_name = self.constants.get("data_model_module")
        if not module_name:
            raise ValueError(
                f"domain.json for '{self.domain}' missing data_model_module"
            )
        module = importlib.import_module(module_name)
        get_db_fn = getattr(module, "get_db")
        return get_db_fn()

    def get_environment(self, db=None, user_db=None):
        """Create an Environment for this domain.

        Args:
            db: Optional DB model instance. If None, uses default empty DB.
            user_db: Optional User DB model instance (telecom only).
        """
        module_name = self.constants.get("environment_module")
        if not module_name:
            raise ValueError(
                f"domain.json for '{self.domain}' missing environment_module"
            )
        fn_name = self.constants.get("environment_function", "get_environment")
        module = importlib.import_module(module_name)
        get_env_fn = getattr(module, fn_name)
        kwargs = {}
        if db is not None:
            kwargs["db"] = db
        if user_db is not None:
            kwargs["user_db"] = user_db
        return get_env_fn(**kwargs)

    def has_user_db(self) -> bool:
        """Whether this domain has a separate user DB."""
        return "user_data_model_module" in self.constants

    def get_user_db_class(self):
        """Dynamically import and return the User DB Pydantic model class.

        Returns None if the domain has no user DB (e.g. airline, retail).
        """
        module_name = self.constants.get("user_data_model_module")
        class_name = self.constants.get("user_db_class")
        if not module_name or not class_name:
            return None
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    def get_user_db(self):
        """Return a default UserDB instance for this domain.

        Returns None if the domain has no user DB.
        """
        cls = self.get_user_db_class()
        if cls is None:
            return None
        return cls()

    def get_db_schema_str(self) -> str:
        """Return a JSON-schema description of this domain's DB models.

        Used to feed the ``generate_db_initialization`` prompt with
        domain-agnostic schema information instead of hard-coding the schema
        per domain.

        Output shape::

            {
              "agent_data": { ...JSON schema for the backend DB... },
              "user_data":  { ...JSON schema for the user DB, if any... }
            }

        ``user_data`` is omitted entirely when the domain has no user DB
        (airline, retail). Schemas are produced via Pydantic's
        ``model_json_schema()``.
        """
        schema: Dict[str, Any] = {
            "agent_data": self.get_db_class().model_json_schema(),
        }
        user_db_class = self.get_user_db_class()
        if user_db_class is not None:
            schema["user_data"] = user_db_class.model_json_schema()
        return json.dumps(schema, indent=2)

    @property
    def action_group_map(self) -> Optional[Dict[str, str]]:
        """Semantic action-group mapping for weighted edit distance.

        Returns the mapping from ``domain.json["action_group_map"]`` if
        present, otherwise ``None`` (falls back to prefix heuristic).
        """
        return self.constants.get("action_group_map")

    def get_domain_validator(self):
        """Return the domain-specific validator instance."""
        from src.common.domain_validators import get_domain_validator
        return get_domain_validator(self)
