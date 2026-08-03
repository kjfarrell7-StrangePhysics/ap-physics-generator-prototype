import json
import re
import matplotlib.pyplot as plt
import numpy as np
import openai
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

# ==========================================
# 1. HELPER FUNCTIONS: LATEX & JSON CLEANUP
# ==========================================

def clean_latex_for_streamlit(text: str) -> str:
    if not text:
        return ""
    text = text.replace(r"\n", "\n")
    text = re.sub(r"[\x08]oldsymbol\{([^}]+)\}", r"\\vec{\1}", text)
    text = re.sub(r"\\boldsymbol\{([^}]+)\}", r"\\vec{\1}", text)
    text = re.sub(r"\\\[\s*", "$$", text)
    text = re.sub(r"\s*\\\]", "$$", text)
    text = re.sub(r"\\\(\s*", "$", text)
    text = re.sub(r"\s*\\\)", "$", text)
    return text

def sanitize_json_response(raw_str: str) -> dict:
    def fix_slash(match):
        return "\\\\"
    cleaned = re.sub(r'\\\\|\\(?!["\\])', fix_slash, raw_str)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        cleaned_fallback = re.sub(r'\\(?!["\\])', r"\\\\", raw_str)
        return json.loads(cleaned_fallback, strict=False)

# ==========================================
# 2. HARDENED DIAGRAM ENGINE
# ==========================================

def safe_float(val, default=1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

def draw_synchronized_diagram(diag_type, vectors, title="Physics Visual Model", extra_params=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    if extra_params is None:
        extra_params = {}

    if diag_type == "graph":
        # Hardened fallback defaults to prevent blank renders
        x_data = extra_params.get("x_data", [0, 1, 2, 3, 4])
        y_data = extra_params.get("y_data", [0, 2.5, 5.0, 7.5, 10.0])
        x_label = extra_params.get("x_label", "Time (s)")
        y_label = extra_params.get("y_label", "Velocity (m/s)")
        shade_area = extra_params.get("shade_area", False)

        curve_type = extra_params.get("curve_type", "linear")
        
        # Plot the main kinematic line
        if curve_type == "quadratic":
            ax.plot(x_data, y_data, color="#0033cc", linewidth=3, linestyle="-", marker="o")
        else:
            ax.plot(x_data, y_data, color="#0033cc", linewidth=3, marker="o")

        # Pedagogical Tool: Shade area under curve if testing displacement/work
        if shade_area:
            ax.fill_between(x_data, y_data, color="#0033cc", alpha=0.2)

        ax.set_xlabel(clean_latex_for_streamlit(x_label), fontsize=11, weight="bold")
        ax.set_ylabel(clean_latex_for_streamlit(y_label), fontsize=11, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        max_x = max(x_data) if x_data else 5
        max_y = max(y_data) if y_data else 10
        min_y = min(y_data) if y_data else 0
        
        # Dynamic axis buffering
        ax.set_xlim(0, max_x * 1.05 if max_x > 0 else 5)
        ax.set_ylim(min_y * 1.1 if min_y < 0 else 0, max_y * 1.15 if max_y > 0 else 10)

    elif diag_type == "incline":
        theta = safe_float(extra_params.get("angle", 30))
        theta_rad = np.radians(theta)
        L = 4.0
        ax.fill([0, L, L, 0], [0, 0, L * np.tan(theta_rad), 0], color="#e0e0e0", edgecolor="black", zorder=1)
        bx, by = L * 0.5, (L * 0.5) * np.tan(theta_rad)
        ax.plot(bx, by, "ks", markersize=14, zorder=3)
        # Vector rendering simplified for brevity...
        ax.set_aspect("equal")
        
    else: # Default Free Body Diagram
        ax.plot(0, 0, "ko", markersize=8)
        # Standard FBD rendering...
        ax.set_aspect("equal")

    ax.set_title(title, fontsize=12, pad=12, weight="bold")
    plt.tight_layout()
    return fig

# ==========================================
# 3. PEDAGOGICALLY ALIGNED GENERATOR
# ==========================================

def generate_single_ap_question(course, q_format, q_style, topic, previous_questions, api_key):
    client = openai.OpenAI(api_key=api_key)

    is_calculus = "C:" in course
    rigor = "CALCULUS-BASED RIGOR (AP Physics C)." if is_calculus else "ALGEBRA-BASED RIGOR (AP Physics 1 & 2)."

    if q_style == "Conceptual / Qualitative":
        style_instruction = (
            "PEDAGOGICAL TARGET (Conceptual): Focus on physics concepts using qualitative tools (graphs, motion maps). "
            "MANDATORY DIAGRAM RULE: 'has_diagram' MUST be true. Set 'diag_type' to 'graph'. "
            "CRITICAL: You MUST provide 'x_data' and 'y_data' arrays of at least 4 coordinate points in 'extra_params'. "
            "If assessing displacement from a v-t graph, set 'shade_area': true in extra_params."
        )
    elif q_style == "Quantitative Arithmetic":
        style_instruction = (
            "PEDAGOGICAL TARGET (Quantitative): Focus on algebraic execution and computation. "
            "MANDATORY DIAGRAM RULE: 'has_diagram' MUST be false. Do NOT give away numerical values on a graph axis."
        )
    else:
        style_instruction = (
            "PEDAGOGICAL TARGET (Experimental Design): Focus on laboratory setups, data collection, or error analysis. "
            "MANDATORY DIAGRAM RULE: 'has_diagram' MUST be true."
        )

    system_prompt = f"""
    You are an expert AP Physics item writer. Generate EXACTLY ONE rigorous {course} question.
    {rigor} {style_instruction}

    JSON STRUCTURE:
    {{
        "scratchpad_derivation": "Solve physics and check distractors here FIRST.",
        "calculated_target_value": "20 m", 
        "question_text": "A car moves with...",
        "options": ["A) 10 m", "B) 20 m", "C) 25 m", "D) 40 m"], 
        "correct_answer": "B", 
        "explanation": "Explanation...",
        "has_diagram": true,
        "diag_type": "graph",
        "vectors": [],
        "extra_params": {{
            "x_data": [0, 1, 2, 3, 4],
            "y_data": [5, 5, 5, 5, 5],
            "x_label": "Time (s)",
            "y_label": "Velocity (m/s)",
            "curve_type": "linear",
            "shade_area": true
        }}
    }}
    CRITICAL: Math must use single $ signs. extra_params MUST be populated if diag_type is 'graph'.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2000,
        temperature=0.6,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Topic: {topic}"}],
    )
    return sanitize_json_response(response.choices[0].message.content)

# [Streamlit UI logic remains exactly the same as the previous iteration]
