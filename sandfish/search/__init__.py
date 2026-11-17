"""Opt-in parameter search: researcher agents, external evaluators and an
append-only trial ledger.

SandFish is the orchestration and audit layer here. It proposes variants,
calls *your* evaluator and records every evaluation. It does not backtest,
price or forecast anything, and nothing in this package runs unless you call
it. See the README section "Parameter search (opt-in)".
"""

from .agents import (
    SEARCH_AGENT_TYPES,
    ExploiterAgent,
    ExplorerAgent,
    ResearcherAgent,
    SearchContext,
    attach_search,
    evaluate_and_record,
    register_search_agents,
)
from .evaluator import (
    ENTRY_POINT_GROUP,
    SPLITS,
    TEST,
    TRAIN,
    EvaluationResult,
    Evaluator,
    FunctionEvaluator,
    ToyQuadraticEvaluator,
    as_evaluator,
    available_evaluators,
    get_evaluator,
    register_evaluator,
    unregister_evaluator,
)
from .ledger import GENESIS, LedgerIntegrityError, TrialLedger, variant_key, verify_ledger_file
from .runner import MULTIPLE_TESTING_NOTE, run_grid, run_search, summarize, validate_holdout
from .space import grid_size, iter_grid, validate_space

__all__ = [
    "ENTRY_POINT_GROUP", "GENESIS", "MULTIPLE_TESTING_NOTE", "SEARCH_AGENT_TYPES", "SPLITS",
    "TEST", "TRAIN", "EvaluationResult", "Evaluator", "ExploiterAgent", "ExplorerAgent",
    "FunctionEvaluator", "LedgerIntegrityError", "ResearcherAgent", "SearchContext",
    "ToyQuadraticEvaluator", "TrialLedger", "as_evaluator", "attach_search",
    "available_evaluators", "evaluate_and_record", "get_evaluator", "grid_size", "iter_grid",
    "register_evaluator", "register_search_agents", "run_grid", "run_search", "summarize",
    "unregister_evaluator", "validate_holdout", "validate_space", "variant_key",
    "verify_ledger_file",
]
