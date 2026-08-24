"""
RiskSense AI - Ollama Service

Communicates with local Ollama via REST API for:
- Semantic interpretation of release changes
- Human-readable risk explanations

The LLM never determines scores, counts, or historical facts.
If Ollama is unavailable, falls back to template-based explanations.
"""

import requests

import config


def check_ollama_status():
    """
    Check if Ollama is running and llama3.2 model is available.

    Returns:
        dict with keys: available (bool), models (list), message (str)
    """
    try:
        resp = requests.get(
            f"{config.OLLAMA_BASE_URL}/api/tags",
            timeout=config.OLLAMA_CONNECT_TIMEOUT,
        )
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            model_available = any(config.OLLAMA_MODEL in m for m in models)
            return {
                "available": True,
                "model_available": model_available,
                "models": models,
                "message": (
                    f"Ollama is running. Model '{config.OLLAMA_MODEL}' "
                    + ("is available." if model_available else "is NOT available.")
                ),
            }
        return {
            "available": False,
            "model_available": False,
            "models": [],
            "message": f"Ollama responded with status {resp.status_code}.",
        }
    except requests.ConnectionError:
        return {
            "available": False,
            "model_available": False,
            "models": [],
            "message": "Ollama is not running. Risk scores remain fully functional.",
        }
    except requests.Timeout:
        return {
            "available": False,
            "model_available": False,
            "models": [],
            "message": "Ollama connection timed out.",
        }
    except Exception as e:
        return {
            "available": False,
            "model_available": False,
            "models": [],
            "message": f"Ollama check failed: {str(e)}",
        }


def generate_risk_explanation(test_info, risk_factors):
    """
    Generate a human-readable risk explanation for a test case.
    Uses Ollama if available, otherwise falls back to template.

    Args:
        test_info: dict with test case details (test_id, test_name, module, etc.)
        risk_factors: dict with sub-scores and context

    Returns:
        str: Human-readable explanation
    """
    status = check_ollama_status()

    if status["available"] and status["model_available"]:
        try:
            return _ollama_risk_explanation(test_info, risk_factors)
        except Exception:
            return _template_risk_explanation(test_info, risk_factors)
    else:
        return _template_risk_explanation(test_info, risk_factors)


def generate_change_impact_analysis(change_info, module):
    """
    Generate semantic analysis of a release change's impact.
    Uses Ollama if available, otherwise falls back to template.

    Args:
        change_info: dict with change details
        module: str, the hospital module

    Returns:
        str: Impact analysis text
    """
    status = check_ollama_status()

    if status["available"] and status["model_available"]:
        try:
            return _ollama_change_analysis(change_info, module)
        except Exception:
            return _template_change_analysis(change_info, module)
    else:
        return _template_change_analysis(change_info, module)


def _ollama_risk_explanation(test_info, risk_factors):
    """Call Ollama for a risk explanation."""
    prompt = f"""You are a QA risk analyst. Provide a brief, professional risk explanation (3-4 sentences) for this test case.

Test Case: {test_info.get('test_name', 'Unknown')}
Module: {test_info.get('module', 'Unknown')}
Risk Score: {risk_factors.get('risk_score', 0)}/100
Risk Level: {risk_factors.get('risk_level', 'Unknown')}

Risk Factors:
- Historical Defect Score: {risk_factors.get('historical_defect_score', 0)}/100 (weight: 25%)
- Historical Failure Score: {risk_factors.get('historical_failure_score', 0)}/100 (weight: 20%)
- Change Impact Score: {risk_factors.get('change_impact_score', 0)}/100 (weight: 25%)
- Business Criticality: {risk_factors.get('business_criticality_score', 0)}/100 (weight: 15%)
- Safety/Regulatory: {risk_factors.get('safety_regulatory_score', 0)}/100 (weight: 15%)

Context:
- Historical Defects: {risk_factors.get('defect_count', 0)}
- Failure Rate: {risk_factors.get('failure_rate', 0)}%
- R5 Changes: {risk_factors.get('change_count', 0)}

Explain WHY this test is rated at this risk level based on the provided scores. Focus on the dominant risk factors. Do not invent any numbers — only reference the values provided above."""

    return _call_ollama(prompt)


def _ollama_change_analysis(change_info, module):
    """Call Ollama for change impact analysis."""
    prompt = f"""You are a QA risk analyst. Provide a brief impact analysis (2-3 sentences) for this release change.

Change: {change_info.get('description', 'Unknown')}
Type: {change_info.get('change_type', 'Unknown')}
Module: {module}
Impact Area: {change_info.get('impact_area', 'Unknown')}

Explain the potential testing impact of this change on the {module} module. Be specific about what areas of testing should receive attention. Do not invent any metrics or data."""

    return _call_ollama(prompt)


def _call_ollama(prompt):
    """Make a generation request to Ollama."""
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 256,
            },
        },
        timeout=(config.OLLAMA_CONNECT_TIMEOUT, config.OLLAMA_READ_TIMEOUT),
    )

    if resp.status_code == 200:
        return resp.json().get("response", "No response generated.")
    else:
        raise RuntimeError(f"Ollama returned status {resp.status_code}")


def _template_risk_explanation(test_info, risk_factors):
    """Generate a template-based risk explanation (fallback)."""
    module = test_info.get("module", "Unknown")
    risk_level = risk_factors.get("risk_level", "Unknown")
    risk_score = risk_factors.get("risk_score", 0)

    # Identify dominant factors
    factors = {
        "Historical Defect Risk": risk_factors.get("historical_defect_score", 0),
        "Historical Failure Risk": risk_factors.get("historical_failure_score", 0),
        "Change Impact": risk_factors.get("change_impact_score", 0),
        "Business Criticality": risk_factors.get("business_criticality_score", 0),
        "Safety/Regulatory": risk_factors.get("safety_regulatory_score", 0),
    }
    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    top_factor = sorted_factors[0]
    second_factor = sorted_factors[1]

    explanation = (
        f"This test case in the {module} module has a {risk_level} risk rating "
        f"with a score of {risk_score}/100. "
        f"The primary risk driver is {top_factor[0]} (score: {top_factor[1]}/100), "
        f"followed by {second_factor[0]} (score: {second_factor[1]}/100). "
    )

    defects = risk_factors.get("defect_count", 0)
    failure_rate = risk_factors.get("failure_rate", 0)
    changes = risk_factors.get("change_count", 0)

    if defects > 0:
        explanation += f"There are {defects} historical defect(s) associated with this test. "
    if failure_rate > 0:
        explanation += f"The historical failure rate is {failure_rate}%. "
    if changes > 0:
        explanation += f"There are {changes} change(s) in the current release affecting this area."

    return explanation


def _template_change_analysis(change_info, module):
    """Generate a template-based change analysis (fallback)."""
    change_type = change_info.get("change_type", "Unknown")
    impact_area = change_info.get("impact_area", "Unknown")
    description = change_info.get("description", "Unknown change")

    impact_notes = {
        "Bug Fix": "This bug fix indicates a known issue that may have broader implications. Regression testing around the fix area is recommended.",
        "New Feature": "New functionality introduces potential for integration issues. Both the new feature and existing workflows should be tested.",
        "Enhancement": "This enhancement modifies existing behavior. Verify that the enhancement works as expected and does not break existing functionality.",
        "Configuration": "Configuration changes can have cascading effects. Verify system behavior under the new configuration.",
        "Refactoring": "Code refactoring should not change external behavior but may introduce subtle issues. Focus testing on the refactored areas.",
    }

    analysis = (
        f"Change: {description}. "
        f"Impact Area: {impact_area}. "
        f"{impact_notes.get(change_type, 'Standard testing is recommended.')}"
    )
    return analysis
