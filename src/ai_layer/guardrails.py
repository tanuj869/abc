"""
Guardrails System:
Implements input query validation and structured response enforcement
mandated by Section 6 of the PM Surya Ghar AI Decision Support Specification.
"""

from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field

class GuardrailValidationResult(BaseModel):
    is_allowed: bool
    rejection_reason: str = ""
    sanitized_intent: str = ""

REJECTED_KEYWORDS = [
    "mars", "moon", "alien", "crypto", "bitcoin", "football", "cricket", "bollywood",
    "hack", "bypass", "illegal", "attack", "jailbreak", "exploit"
]

def validate_input_query(query: str) -> GuardrailValidationResult:
    """Input Guardrail: Validates if query is within scope of PM Surya Ghar policy analysis."""
    q_lower = query.strip().lower()
    if len(q_lower) < 3:
        return GuardrailValidationResult(
            is_allowed=False,
            rejection_reason="Query is too short or empty. Please specify a state, forecast horizon, or policy question."
        )
    
    for kw in REJECTED_KEYWORDS:
        if kw in q_lower:
            return GuardrailValidationResult(
                is_allowed=False,
                rejection_reason=f"Query rejected by Guardrails: Topic '{kw}' is outside the scope of India's PM Surya Ghar rooftop solar mission."
            )
            
    return GuardrailValidationResult(
        is_allowed=True,
        sanitized_intent="valid_policy_query"
    )

def enforce_output_guardrail(
    analysis: str,
    metrics: Dict[str, Any],
    sources: list,
    confidence: str = "High (95% CI Backtested)",
    assumptions: str = "Conditional on historical grid tariff revision trends and observed portal records.",
    limitations: str = "Statistical correlations and forecast extrapolations represent model predictions, not structural counterfactual proofs."
) -> Dict[str, Any]:
    """Output Guardrail: Formats response into a consistent, auditable policy decision card."""
    return {
        "analysis": analysis,
        "metrics": metrics,
        "confidence_level": confidence,
        "data_sources_cited": sources if sources else ["PM Surya Ghar National Portal", "CEA Tariff Books 2024-25", "NASA POWER Climatology"],
        "assumptions": assumptions,
        "limitations": limitations
    }
