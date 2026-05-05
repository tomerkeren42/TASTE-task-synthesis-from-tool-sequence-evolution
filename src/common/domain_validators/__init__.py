"""Domain-specific validators."""

from src.common.domain_validators.base import BaseDomainValidator

_VALIDATORS = {
    "airline": "src.common.domain_validators.airline:AirlineValidator",
    "retail": "src.common.domain_validators.retail:RetailValidator",
    "telecom": "src.common.domain_validators.telecom:TelecomValidator",
}


def get_domain_validator(domain_config) -> BaseDomainValidator:
    domain = domain_config.domain
    entry = _VALIDATORS.get(domain)
    if entry is None:
        raise ValueError(f"No validator for domain '{domain}'. Available: {list(_VALIDATORS.keys())}")
    module_path, class_name = entry.rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(domain_config)
