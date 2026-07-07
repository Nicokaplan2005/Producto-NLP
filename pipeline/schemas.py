"""Pydantic schemas shared by the repo-state feature pipeline."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

FeatureValue = str | int | float | bool | None


class ChangeIntent(StrEnum):
    BUG_FIX = "bug_fix"
    HOTFIX = "hotfix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    MIGRATION = "migration"
    CONFIG = "config"
    CLEANUP = "cleanup"


class IntentMatch(StrEnum):
    MATCH = "match"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    UNCLEAR = "unclear"


class YesNoUnknown(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class SemanticRiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskDomain(StrEnum):
    SECURITY = "security"
    AUTH = "auth"
    PAYMENTS = "payments"
    DATA_INTEGRITY = "data_integrity"
    PERFORMANCE = "performance"
    CONCURRENCY = "concurrency"
    API_CONTRACT = "api_contract"
    PRIVACY = "privacy"
    OBSERVABILITY = "observability"
    CONFIGURATION = "configuration"


class BackwardCompatibilityRisk(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImplementationCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    SUPERFICIAL = "superficial"
    UNKNOWN = "unknown"


class ErrorPathHandling(StrEnum):
    NONE = "none"
    WEAK = "weak"
    ADEQUATE = "adequate"
    THOROUGH = "thorough"


class MissingCase(StrEnum):
    NULL_HANDLING = "null_handling"
    EMPTY_INPUT = "empty_input"
    PERMISSIONS_CHECK = "permissions_check"
    CONCURRENCY = "concurrency"
    TIMEOUT = "timeout"
    RETRY_LOGIC = "retry_logic"
    PARTIAL_FAILURE = "partial_failure"
    DATA_VALIDATION = "data_validation"
    ROLLBACK = "rollback"
    MIGRATION_EDGE_CASE = "migration_edge_case"


class TestSemanticRelevance(StrEnum):
    NONE = "none"
    WEAK = "weak"
    PARTIAL = "partial"
    STRONG = "strong"


class MissingRegressionTest(StrEnum):
    TRUE = "true"
    FALSE = "false"
    NOT_APPLICABLE = "not_applicable"


class CouplingRiskSemantic(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AbstractionLevelFit(StrEnum):
    TOO_LOW = "too_low"
    APPROPRIATE = "appropriate"
    TOO_HIGH = "too_high"


class ContextualAdaptationLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChangeScope(StrEnum):
    FOCUSED = "focused"
    CROSS_MODULE = "cross_module"
    BROAD = "broad"


class EnhancedPRFeatures(BaseModel):
    """LLM-extracted semantic features from ``features_mejoradas.csv``."""

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    inferred_change_intent: ChangeIntent = Field(
        description="Main intent: bug_fix, hotfix, feature, refactor, test, docs, migration, config, or cleanup."
    )
    stated_vs_actual_intent_match: IntentMatch = Field(
        description="Whether PR title/body and the actual diff match, partially match, mismatch, or are unclear."
    )
    mixed_concerns: bool = Field(
        description="True when the PR mixes unrelated objectives such as bug fix plus refactor or feature plus style churn."
    )
    diff_addresses_stated_problem: YesNoUnknown = Field(
        description="Whether the code diff actually addresses the underlying stated problem, not just whether it matches the description."
    )
    unexplained_changes_present: bool = Field(
        description="True when the diff contains changes not justified by the stated objective."
    )
    semantic_risk_level: SemanticRiskLevel = Field(
        description="Global semantic merge risk considering touched domains, reversibility, blast radius, and other risk signals."
    )
    risk_domains: list[RiskDomain] = Field(
        default_factory=list,
        description="Closed multi-hot labels for sensitive domains affected by the PR.",
    )
    affects_api_contract: bool = Field(
        description="True when public interfaces, endpoints, schemas, SDK exports, config formats, events, or expected contracts change."
    )
    backward_compatibility_risk: BackwardCompatibilityRisk = Field(
        description="Risk that existing behavior, persisted/transmitted formats, defaults, errors, configs, or integrations break."
    )
    breaks_existing_assumption: YesNoUnknown = Field(
        description="Whether the change invalidates implicit invariants, preconditions, ordering assumptions, hardcoded defaults, roles, or state assumptions."
    )
    security_sensitive_change: bool = Field(
        description="True when authentication, authorization, permissions, secrets, user input validation, access control, sensitive logging, or data exposure changes."
    )
    implementation_completeness: ImplementationCompleteness = Field(
        description="Whether the implementation appears complete, partial, superficial, or unknown relative to the stated problem."
    )
    error_path_handling: ErrorPathHandling = Field(
        description="Quality of production-code handling for nulls, invalid inputs, IO/network errors, external dependency failures, and invalid system states."
    )
    incomplete_integration: bool = Field(
        description="True when the PR introduces a partial integration: connects to an external service, component, or system without completing the full integration contract (e.g. only happy path, missing callbacks, dangling integration points, side-effects on consumers not handled)."
    )
    likely_missing_cases: list[MissingCase] = Field(
        default_factory=list,
        description="Closed multi-hot labels for relevant scenarios that the implementation appears to miss.",
    )
    test_semantic_relevance: TestSemanticRelevance = Field(
        description="Whether added or changed tests semantically verify the changed behavior."
    )
    missing_regression_test: MissingRegressionTest = Field(
        description="For bug_fix/hotfix only: whether a regression test is missing. Use not_applicable for non-bug changes."
    )
    missing_edge_case_tests: bool = Field(
        description="True when tests omit relevant edge cases, boundary inputs, or error conditions."
    )
    coupling_risk_semantic: CouplingRiskSemantic = Field(
        description="Risk that the solution increases coupling between previously separate components."
    )
    abstraction_level_fit: AbstractionLevelFit = Field(
        description="Whether the solution is implemented too low, appropriately, or too high in the repo's abstraction stack."
    )
    follows_existing_repo_patterns: YesNoUnknown = Field(
        description="Whether the implementation follows established repo patterns, helpers, abstractions, error handling, and test style."
    )
    reinvents_existing_functionality: YesNoUnknown = Field(
        description="Whether the PR recreates logic, helpers, services, utilities, or abstractions that already exist in the repo."
    )
    missing_update_to_related_files: YesNoUnknown = Field(
        description="Whether related tests, docs, config, schemas, migrations, or callers probably should have been updated but were not."
    )
    lack_of_contextual_adaptation: ContextualAdaptationLevel = Field(
        description="Degree to which the solution ignores repo-specific conventions, constraints, patterns, architecture, or domain context."
    )
    change_scope: ChangeScope = Field(
        description="Breadth of the change across the repo's module structure. Use the card's modules and module_boundaries: focused = single module/component; cross_module = 2-3 modules or explicitly crosses a boundary; broad = 4+ modules or touches core infrastructure."
    )
    touches_high_risk_area: bool = Field(
        description="True when any changed file path overlaps with paths listed in the repo card's risk_model.high_risk_areas."
    )


class PullRequestRefModel(BaseModel):
    """Normalized reference parsed from a GitHub PR URL."""

    owner: str
    repo: str
    number: int
    original_url: str
    extra_path: tuple[str, ...] = ()

    @property
    def repo_full_name(self) -> str:
        return f"{self.owner}/{self.repo}"


class PullRequestMetadata(BaseModel):
    """GitHub metadata needed to align a local repo to a PR boundary."""

    ref: PullRequestRefModel
    api_url: str
    html_url: str
    diff_url: str
    patch_url: str
    raw_diff_url: str
    base_sha: str
    head_sha: str
    merge_commit_sha: str | None = None
    merged_at: str | None = None
    base_ref: str | None = None
    head_ref: str | None = None
    clone_url: str
    default_branch: str | None = None
    repo_state_before_pr_sha: str
    repo_state_sha_source: str
    raw: dict[str, Any] = Field(default_factory=dict)


class RepoCard(BaseModel):
    """Flexible schema for the global repo card.

    The card has a stable top-level contract, while each section stays flexible
    because the exact content will be produced and evolved by an LLM.
    """

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.3"
    repo: dict[str, Any] = Field(default_factory=dict)
    architecture: dict[str, Any] = Field(default_factory=dict)
    modules: dict[str, Any] = Field(default_factory=dict)
    path_semantics: list[dict[str, Any]] = Field(default_factory=list)
    entrypoints: dict[str, Any] = Field(default_factory=dict)
    public_surfaces: list[Any] = Field(default_factory=list)
    conventions: dict[str, Any] = Field(default_factory=dict)
    key_invariants: list[str] = Field(default_factory=list)
    known_antipatterns: list[str] = Field(default_factory=list)
    utilities_index: list[Any] = Field(default_factory=list)
    module_boundaries: list[Any] = Field(default_factory=list)
    special_files: dict[str, Any] = Field(default_factory=dict)
    risk_model: dict[str, Any] = Field(default_factory=dict)
    navigation: dict[str, Any] = Field(default_factory=dict)
    cli_hints: dict[str, Any] = Field(default_factory=dict)
    update_policy: dict[str, Any] = Field(default_factory=dict)
    maintenance: dict[str, Any] = Field(default_factory=dict)


class CardPatch(BaseModel):
    """Partial card update returned by the LLM — only the top-level sections that changed.

    Fields absent (or None) mean "no change to this section".
    Unknown field names are silently ignored so a bad LLM field name never aborts the run.
    After merging, the result is re-validated as a full RepoCard.
    """

    model_config = ConfigDict(extra="ignore")

    repo: dict[str, Any] | None = None
    architecture: dict[str, Any] | None = None
    modules: dict[str, Any] | None = None
    path_semantics: list[Any] | None = None
    entrypoints: dict[str, Any] | None = None
    public_surfaces: list[Any] | None = None
    conventions: dict[str, Any] | None = None
    key_invariants: list[str] | None = None
    known_antipatterns: list[str] | None = None
    utilities_index: list[Any] | None = None
    module_boundaries: list[Any] | None = None
    special_files: dict[str, Any] | None = None
    risk_model: dict[str, Any] | None = None
    navigation: dict[str, Any] | None = None
    cli_hints: dict[str, Any] | None = None
    update_policy: dict[str, Any] | None = None
    maintenance: dict[str, Any] | None = None


class PRProcessingOutput(BaseModel):
    """Output of a single-call PR processing: features + optional patch (for merged PRs)."""

    model_config = ConfigDict(extra="ignore")

    features: EnhancedPRFeatures
    card_patch: CardPatch | None = None


class ChangedFileFeature(BaseModel):
    path: str
    old_path: str | None = None
    status: str = "modified"
    additions: int = 0
    deletions: int = 0


class ExtractedFeature(BaseModel):
    name: str
    value: FeatureValue
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class FeatureExtractionResult(BaseModel):
    schema_version: str = "1.0"
    pr_url: str
    repo: str
    pr_number: int
    repo_state_sha: str | None = None
    diff_path: str
    card_path: str
    changed_files: list[ChangedFileFeature] = Field(default_factory=list)
    selected_files_to_inspect: list[str] = Field(default_factory=list)
    summary_features: dict[str, FeatureValue] = Field(default_factory=dict)
    semantic_features: EnhancedPRFeatures | None = None
    llm_features: list[ExtractedFeature] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
