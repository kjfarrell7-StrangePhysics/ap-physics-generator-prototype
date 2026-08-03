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
        x_data = extra_params.get("x_data", [0, 1, 2, 3, 4])
        y_data = extra_params.get("y_data", [0, 2.5, 5.0, 7.5, 10.0])
        x_label = extra_params.get("x_label", "Time (s)")
        y_label = extra_params.get("y_label", "Velocity (m/s)")
        shade_area = extra_params.get("shade_area", False)
        curve_type = extra_params.get("curve_type", "linear")
        
        if curve_type == "quadratic":
            ax.plot(x_data, y_data, color="#0033cc", linewidth=3, linestyle="-", marker="o")
        else:
            ax.plot(x_data, y_data, color="#0033cc", linewidth=3, marker="o")

        if shade_area:
            ax.fill_between(x_data, y_data, color="#0033cc", alpha=0.2)

        ax.set_xlabel(clean_latex_for_streamlit(x_label), fontsize=11, weight="bold")
        ax.set_ylabel(clean_latex_for_streamlit(y_label), fontsize=11, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        max_x = max(x_data) if x_data else 5
        max_y = max(y_data) if y_data else 10
        min_y = min(y_data) if y_data else 0
        
        ax.set_xlim(0, max_x * 1.05 if max_x > 0 else 5)
        ax.set_ylim(min_y * 1.1 if min_y < 0 else 0, max_y * 1.15 if max_y > 0 else 10)

    elif diag_type == "incline":
        theta = safe_float(extra_params.get("angle", 30))
        theta_rad = np.radians(theta)
        L = 4.0
        ax.fill([0, L, L, 0], [0, 0, L * np.tan(theta_rad), 0], color="#e0e0e0", edgecolor="black", zorder=1)
        bx, by = L * 0.5, (L * 0.5) * np.tan(theta_rad)
        ax.plot(bx, by, "ks", markersize=14, zorder=3)
        ax.set_aspect("equal")
        
    else:
        ax.plot(0, 0, "ko", markersize=8)
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

# ==========================================
# 4. STREAMLIT UI SETUP (COMPLETE & RESTORED)
# ==========================================

st.set_page_config(page_title="AP Physics Exam Builder", page_icon="⚡", layout="centered")
st.title("⚡ AP Physics Test Builder & Item Generator")

st.markdown("""
*Build your custom exam **one rigorous question at a time**. Each generated item is uniquely crafted, 
pedagogically verified, and paired with custom-rendered visual models designed for professional testing.*
""")

st.subheader("Exam & Target Settings")

col1, col2 = st.columns(2)

with col1:
    course = st.selectbox("AP Course", [
        "AP Physics 1", 
        "AP Physics 2", 
        "AP Physics C: Mechanics", 
        "AP Physics C: Electricity & Magnetism"
    ])
    
    q_format = st.selectbox("Question Format", [
        "Single-Select Multiple Choice (MC)",
        "Multi-Select Multiple Choice (MS)",
        "Free Response Question (FRQ)"
    ])

with col2:
    q_style = st.selectbox("Pedagogical Question Style / Category", [
        "Conceptual / Qualitative",
        "Quantitative Arithmetic",
        "Experimental Design & Analytical Skills"
    ])

    ap_topics_map = {
        "AP Physics 1": ["Kinematics", "Dynamics", "Circular Motion and Gravitation", "Work, Energy, and Power", "Linear Momentum", "Torque and Rotational Motion", "Simple Harmonic Motion", "Fluids"],
        "AP Physics 2": ["Fluids", "Thermodynamics", "Electric Force, Field, and Potential", "Electric Circuits", "Magnetism and Electromagnetism", "Optics", "Modern Physics"],
        "AP Physics C: Mechanics": ["Kinematics", "Newton's Laws of Motion", "Work, Energy, and Power", "Systems of Particles and Linear Momentum", "Rotation and Angular Momentum", "Oscillations", "Gravitation"],
        "AP Physics C: Electricity & Magnetism": ["Electrostatics", "Conductors, Capacitors, and Dielectrics", "Electric Circuits", "Magnetic Fields", "Electromagnetism"]
    }

    available_topics = ap_topics_map.get(course, ["General Physics"])
    topic = st.selectbox("Physics Topic", available_topics)

if "test_bank" not in st.session_state:
    st.session_state["test_bank"] = []

generate_clicked = st.button("Generate Next Test Question", type="primary")

if generate_clicked:
    with st.spinner("Synthesizing unique AP item and enforcing pedagogical constraints..."):
        try:
            prev_questions_context = [
                {"question_text": q.get("question_text", ""), "diag_type": q.get("diag_type", "none")} 
                for q in st.session_state["test_bank"]
            ]
            
            new_q = generate_single_ap_question(course, q_format, q_style, topic, prev_questions_context, st.secrets["OPENAI_API_KEY"])
            if new_q and "question_text" in new_q:
                st.session_state["test_bank"].append(new_q)
            else:
                st.error("Failed to parse valid question structure. Please retry.")
        except Exception as e:
            st.error(f"Error generating question: {e}")

# --- RENDER CURRENTLY BUILT TEST BANK ---
if st.session_state["test_bank"]:
    st.divider()
    st.header(f"📋 Current Exam Draft ({course} — {topic})")
    st.markdown(f"*Total Questions in Test Bank: {len(st.session_state['test_bank'])}*")

    for i, q in enumerate(st.session_state["test_bank"], start=1):
        st.markdown(f"---")
        q_clean = clean_latex_for_streamlit(q.get("question_text", ""))
        st.markdown(f"### Question {i}\n{q_clean}")

        if q.get("has_diagram", False) and q.get("diag_type", "none") != "none":
            diag_type = q.get("diag_type", "graph")
            vectors = q.get("vectors", [])
            extra_params = q.get("extra_params", {})
            fig = draw_synchronized_diagram(diag_type, vectors, title=f"Q{i} Visual Model: {topic}", extra_params=extra_params)
            st.pyplot(fig)

        options = [clean_latex_for_streamlit(opt) for opt in q.get("options", [])]

        if options and q_format != "Free Response Question (FRQ)":
            st.markdown(f"#### Choose Your Answer (Q{i}):")
            
            if "Multi-Select" in q_format:
                user_selections = []
                for idx, opt in enumerate(options):
                    if st.checkbox(opt, key=f"q_{i}_ms_opt_{idx}"):
                        user_selections.append(opt[0])
                
                if st.button(f"Check Answer (Q{i})", key=f"check_btn_{i}"):
                    raw_correct = str(q.get("correct_answer", ""))
                    correct_letters = [c.strip() for c in re.split(r'[,&]', raw_correct) if c.strip() in "ABCD"]
                    
                    if sorted(user_selections) == sorted(correct_letters):
                        st.success(f"🎉 Q{i} Correct! Answers: {', '.join(correct_letters)}.")
                    else:
                        st.error(f"❌ Q{i} Incorrect. Correct answers: {', '.join(correct_letters)}.")
            else:
                user_choice = st.radio("Options", options, key=f"q_{i}_radio", index=0)
                
                if st.button(f"Check Answer (Q{i})", key=f"check_btn_{i}"):
                    correct_letter = str(q.get("correct_answer", "")).strip()
                    if user_choice.startswith(correct_letter):
                        st.success(f"🎉 Q{i} Correct! Answer: {correct_letter}.")
                    else:
                        st.error(f"❌ Q{i} Incorrect. Correct answer: {correct_letter}.")

        with st.expander(f"View Solution & Explanation — Question {i}"):
            st.markdown(f"**Correct Answer:** {q.get('correct_answer', 'N/A')}")
            st.markdown(clean_latex_for_streamlit(q.get("explanation", "")))

    if st.button("Clear Test Bank / Start New Test"):
        st.session_state["test_bank"] = []
        st.rerun()

    # --- EXAM FEEDBACK LOGGING ---
    st.divider()
    st.subheader("Provide Feedback on Question Quality")
    rating = st.radio("Rating", ["👍 Good", "👎 Needs Improvement"], key="exam_rating")
    comment = st.text_area("Comments / Bug Details for this Test", key="exam_comment")

    if st.button("Submit Test Feedback", key="submit_exam_feedback"):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            spreadsheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
            existing_df = conn.read(spreadsheet=spreadsheet_url)
            
            new_row = {
                "Timestamp": str(np.datetime64("now")),
                "Course": course,
                "Format": q_format,
                "Style": q_style,
                "Topic": topic,
                "QuestionCount": len(st.session_state["test_bank"]),
                "Rating": rating,
                "Comment": comment
            }
            
            new_row_df = pd.DataFrame([new_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
            conn.update(spreadsheet=spreadsheet_url, data=updated_df)
            st.success("Test feedback successfully saved to Google Sheets!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
