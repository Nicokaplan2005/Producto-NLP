"""Shared card helpers extracted for reuse across pipeline stages."""

from __future__ import annotations

from .schemas import RepoCard


def default_repo_card() -> RepoCard:
    """Return a skeleton RepoCard in compact v1.3 shape."""
    return RepoCard.model_validate({
        "schema_version": "1.3",
        "repo": {
            "name": "",
            "url": "",
            "default_branch": "",
            "last_verified_commit": "",
            "repo_type": "",
            "primary_languages": [],
        },
        "architecture": {
            "one_liner": "",
            "abstraction_layers": [],
            "main_components": [],
            "data_flow": "",
            "external_services": [],
            "important_runtime_entrypoints": [],
        },
        "modules": {},
        "path_semantics": [],
        "entrypoints": {
            "routes_or_views": [],
            "cli": [],
            "library_exports": [],
            "background_jobs": [],
            "scheduled_jobs": [],
            "event_handlers": [],
            "plugin_interfaces": [],
        },
        "public_surfaces": [],
        "conventions": {
            "testing": "",
            "error_handling": "",
            "logging": "",
            "naming": "",
            "imports": "",
            "patterns": [],
        },
        "key_invariants": [],
        "known_antipatterns": [],
        "utilities_index": [],
        "module_boundaries": [],
        "special_files": {
            "dependency_manifests": [],
            "lockfiles": [],
            "ci_files": [],
            "docker_files": [],
            "deployment_files": [],
            "security_sensitive_files": [],
            "database_files": [],
            "generated_files": [],
            "documentation_files": [],
            "license_files": [],
        },
        "risk_model": {
            "high_risk_areas": [],
            "medium_risk_areas": [],
            "low_risk_areas": [],
        },
        "navigation": {
            "related_file_rules": [],
            "integration_hotspots": [],
            "security_hotspots": [],
            "test_entry": "",
        },
        "cli_hints": {},
        "update_policy": {},
        "maintenance": {
            "created_from_commit": "",
            "last_updated_from_pr": None,
            "known_stale_sections": [],
            "confidence_notes": [],
        },
    })
