"""
LangGraph Multi-Stage Workflow Orchestrator:
Implements stateful workflow graph managing data retrieval, model execution,
scenario simulation, Vectorless RAG, and Guardrail enforcement.
"""

import os
import sys
import re

# Ensure workspace root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from src.ai_layer.guardrails import validate_input_query, enforce_output_guardrail
from src.ai_layer.tools import (
    get_state_metrics, compare_states, get_adoption_forecast,
    run_scenario_simulation, search_policy_documents
)

class AgentState(TypedDict):
    query: str
    is_valid: bool
    rejection_reason: Optional[str]
    intent: str
    target_state: Optional[str]
    target_state_b: Optional[str]
    tool_outputs: Dict[str, Any]
    rag_passages: List[Dict[str, Any]]
    final_response: Dict[str, Any]

# --- NODE 1: INPUT GUARDRAIL ---
def guardrail_node(state: AgentState) -> AgentState:
    val = validate_input_query(state["query"])
    if not val.is_allowed:
        return {
            **state,
            "is_valid": False,
            "rejection_reason": val.rejection_reason,
            "final_response": {
                "analysis": f"❌ {val.rejection_reason}",
                "metrics": {},
                "confidence_level": "N/A",
                "data_sources_cited": [],
                "assumptions": "Query rejected prior to model dispatch.",
                "limitations": "Out-of-scope filter triggered."
            }
        }
    return {**state, "is_valid": True, "rejection_reason": None}

# --- NODE 2: INTENT ROUTER ---
def router_node(state: AgentState) -> AgentState:
    q = state["query"].lower()
    
    # Check for comparison
    comp_match = re.search(r'compare\s+([a-zA-Z\s]+)\s+(?:and|with|to|vs)\s+([a-zA-Z\s]+)', q)
    if "compare" in q or " vs " in q or comp_match:
        # Extract states
        st_names = [
            "gujarat", "maharashtra", "rajasthan", "tamil nadu", "uttar pradesh",
            "kerala", "bihar", "karnataka", "andhra pradesh", "madhya pradesh",
            "odisha", "assam", "haryana", "punjab", "west bengal", "delhi"
        ]
        found = [s.title() for s in st_names if s in q]
        st_a = found[0] if len(found) > 0 else "Gujarat"
        st_b = found[1] if len(found) > 1 else "Maharashtra"
        return {**state, "intent": "compare", "target_state": st_a, "target_state_b": st_b}
        
    # Check for forecast / future projection
    if any(w in q for w in ["forecast", "predict", "projection", "future", "next 12 months", "next 6 months", "horizon"]):
        return {**state, "intent": "forecast"}
        
    # Check for scenario / what-if / tariff shock
    if any(w in q for w in ["what if", "scenario", "shock", "increase tariff", "raise tariff", "subsidy increase"]):
        return {**state, "intent": "simulation"}
        
    # Check for specific state query
    st_names = [
        "gujarat", "maharashtra", "rajasthan", "tamil nadu", "uttar pradesh",
        "kerala", "bihar", "karnataka", "andhra pradesh", "madhya pradesh",
        "odisha", "assam", "haryana", "punjab", "west bengal", "delhi", "mizoram", "goa", "ladakh"
    ]
    found = [s.title() for s in st_names if s in q]
    if found:
        return {**state, "intent": "state_deepdive", "target_state": found[0]}
        
    # Default to policy RAG search
    return {**state, "intent": "rag_search"}

# --- NODE 3: TOOL EXECUTION & DATA RETRIEVAL ---
def execution_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    tool_out = {}
    rag_res = []
    
    if intent == "compare":
        tool_out = compare_states.invoke({
            "state_a": state.get("target_state") or "Gujarat",
            "state_b": state.get("target_state_b") or "Maharashtra"
        })
    elif intent == "forecast":
        scope = state.get("target_state") or "national"
        tool_out = get_adoption_forecast.invoke({"scope": scope})
    elif intent == "simulation":
        # Extract tariff percentage if mentioned
        match = re.search(r'(\d+)\s*%', state["query"])
        t_val = float(match.group(1)) if match else 10.0
        tool_out = run_scenario_simulation.invoke({"tariff_shock_pct": t_val, "target_high_potential": True})
    elif intent == "state_deepdive":
        st_name = state.get("target_state") or "Gujarat"
        tool_out = get_state_metrics.invoke({"state_name": st_name})
        rag_res = search_policy_documents.invoke({"query": f"{st_name} rooftop solar adoption tariff"})["retrieved_passages"]
    else: # rag_search
        rag_res = search_policy_documents.invoke({"query": state["query"]})["retrieved_passages"]
        tool_out = {"passages_retrieved": len(rag_res)}
        
    return {**state, "tool_outputs": tool_out, "rag_passages": rag_res}

# --- NODE 4: SYNTHESIS & REASONING NODE ---
def synthesis_node(state: AgentState) -> AgentState:
    intent = state["intent"]
    tool_out = state["tool_outputs"]
    rag_p = state.get("rag_passages", [])
    
    # Try calling OpenAI if API key works
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and len(openai_key.strip()) > 10:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=600)
            sys_prompt = (
                "You are the SuryaIQ PM Surya Ghar Policy Decision Support Assistant for India's 10-Million Rooftop Solar Mission. "
                "Base your response strictly on the retrieved data and evidence provided below. "
                "Be concise, quantitative, and professional. Structure your answer with clear bullet points."
            )
            context = f"Query Intent: {intent}\nTool Data: {tool_out}\nRetrieved Passages: {rag_p}"
            response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=f"User Query: {state['query']}\n\nEvidence Context:\n{context}")])
            analysis_text = response.content
            
            final_card = enforce_output_guardrail(
                analysis=analysis_text,
                metrics=tool_out,
                sources=[p["doc_name"] for p in rag_p] if rag_p else ["PM Surya Ghar Administrative Panel", "CEA Tariff Books 2024-25"]
            )
            return {**state, "final_response": final_card}
        except Exception:
            # Fall back to deterministic grounded synthesis if network or quota issue
            pass
            
    # Deterministic Expert Synthesis Engine (offline-resilient)
    if intent == "compare":
        c = tool_out.get("comparison", {})
        sa = state.get("target_state", "State A")
        sb = state.get("target_state_b", "State B")
        analysis = (
            f"### Comparative Evaluation: {sa} vs. {sb}\n\n"
            f"• **Volume Scale:** {tool_out.get('higher_volume_state')} leads in aggregate installations.\n"
            f"• **Operational Efficiency:** {tool_out.get('higher_conversion_state')} achieves higher conversion efficiency from sanction applications.\n"
            f"• **Economic Driver:** Differences in recent grid electricity tariff growth explain significant variance in adoption momentum between these states."
        )
    elif intent == "forecast":
        if "horizons" in tool_out:
            h_lines = "\n".join([f"• **{h['horizon']}:** {h['point_forecast']:,} systems (95% CI: {h['confidence_interval_95']})" for h in tool_out["horizons"]])
            analysis = (
                f"### National Adoption Projections (+3, +6, +12 Months)\n\n"
                f"Using the verified **Log-Linear Growth Model (4.67% backtested MAPE)**:\n\n{h_lines}\n\n"
                f"Adoption follows strong compounding dynamics during this expansion phase, pointing towards multi-million scaling over the next 12 months."
            )
        else:
            analysis = f"### State Adoption Projections for {tool_out.get('scope')}\n\n" + str(tool_out)
    elif intent == "simulation":
        imp = tool_out.get("national_impact", {})
        top_b = tool_out.get("top_beneficiary_states", [])
        b_lines = ", ".join([f"{b['state']} (+{b['incremental_units']:,})" for b in top_b[:3]])
        analysis = (
            f"### Counterfactual Policy Simulation Results\n\n"
            f"• **Simulated Policy Shock:** {tool_out.get('inputs', {}).get('tariff_shock_applied')} tariff escalation with sunbelt prioritization.\n"
            f"• **Projected Adoption Lift:** **+{imp.get('incremental_installations', 0):,}** incremental installations nationwide.\n"
            f"• **Grid Impact:** **+{imp.get('incremental_capacity_mw', 0):,} MW** additional solar capacity generating ~**{imp.get('estimated_annual_clean_power_gwh', 0):,} GWh/year**.\n"
            f"• **Top Beneficiary States:** {b_lines}."
        )
    elif intent == "state_deepdive":
        st_data = tool_out
        analysis = (
            f"### Operational Profile: {st_data.get('state')} ({st_data.get('region')} Region)\n\n"
            f"• **Installations:** **{st_data.get('installations_aug2026', 0):,}** completed rooftop systems.\n"
            f"• **Conversion Ratio:** **{st_data.get('conversion_ratio', 0)*100:.1f}%** ({st_data.get('efficiency_tier')}).\n"
            f"• **Per-Capita Rate:** **{st_data.get('installations_per_lakh_pop')}** installations per lakh population.\n"
            f"• **Grid Economics:** Tariff growth of **{st_data.get('tariff_growth_pct'):+.1f}% YoY** with mean daily insolation of **{st_data.get('irradiance_annual_kwh_m2_day')} kWh/m²/day**."
        )
    else:
        passages = [f"• *[{p['doc_name']}]:* {p['text']}" for p in rag_p[:2]]
        analysis = (
            f"### Policy & Research Evidence Synthesis\n\n"
            f"Retrieved key evidence from project documentation and guidelines:\n\n" + "\n\n".join(passages)
        )
        
    final_card = enforce_output_guardrail(
        analysis=analysis,
        metrics=tool_out,
        sources=[p["doc_name"] for p in rag_p] if rag_p else ["PM Surya Ghar Canonical Panel", "CEA Tariff Books 2024-25"]
    )
    return {**state, "final_response": final_card}

# --- BUILD LANGGRAPH WORKFLOW ---
def create_ai_workflow():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("router", router_node)
    workflow.add_node("execution", execution_node)
    workflow.add_node("synthesis", synthesis_node)
    
    workflow.set_entry_point("guardrail")
    
    # Conditional edge: if invalid query, stop; else route
    def route_after_guardrail(state: AgentState):
        return END if not state.get("is_valid", True) else "router"
        
    workflow.add_conditional_edges("guardrail", route_after_guardrail, {END: END, "router": "router"})
    workflow.add_edge("router", "execution")
    workflow.add_edge("execution", "synthesis")
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()

# Singleton graph app
ai_app = create_ai_workflow()

def run_policy_assistant(query: str) -> Dict[str, Any]:
    """Execute the end-to-end LangGraph decision support workflow for a user prompt."""
    res = run_policy_assistant_with_trace(query)
    return res.get("final_response", {})

def run_policy_assistant_with_trace(query: str) -> Dict[str, Any]:
    """Execute the end-to-end workflow and return full execution trace for auditing and UI."""
    init_state: AgentState = {
        "query": query,
        "is_valid": True,
        "rejection_reason": None,
        "intent": "unknown",
        "target_state": None,
        "target_state_b": None,
        "tool_outputs": {},
        "rag_passages": [],
        "final_response": {}
    }
    result = ai_app.invoke(init_state)
    return result

