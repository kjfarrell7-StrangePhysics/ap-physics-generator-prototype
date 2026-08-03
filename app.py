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
    """Converts LaTeX bracket/raw delimiters to standard Streamlit dollar-sign delimiters,

    cleans backspace artifacts, and fixes literal double-escaped \\n sequences.
    """
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
    """Robustly converts raw single LaTeX backslashes in GPT output into double backslashes

    so json.loads() won't crash on invalid escape sequences.
    """

    def fix_slash(match):
        group = match.group(0)
        if group == "\\\\":
            return "\\\\"
        return "\\\\"

    cleaned = re.sub(r'\\\\|\\(?!["\\])', fix_slash, raw_str)

    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        cleaned_fallback = re.sub(r'\\(?!["\\])', r"\\\\", raw_str)
        return json.loads(cleaned_fallback, strict=False)


# ==========================================
# 2. FULLY SYNCHRONIZED & POLISHED DIAGRAM ENGINE
# ==========================================


def safe_float(val, default=1.0):
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def draw_synchronized_diagram(
    diag_type, vectors, title="Physics Visual Model", extra_params=None
):
    """Renders graphics driven directly by the AI's calculated variables, with

    clean, publication-grade axis padding and framing for professional display.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    if extra_params is None:
        extra_params = {}

    if diag_type == "incline":
        theta = safe_float(extra_params.get("angle", 30))
        theta_rad = np.radians(theta)
        L = 4.0
        x_vals = [0, L, L, 0]
        y_vals = [0, 0, L * np.tan(theta_rad), 0]
        ax.fill(
            x_vals,
            y_vals,
            color="#e0e0e0",
            edgecolor="black",
            linewidth=1.5,
            zorder=1,
        )

        bx = L * 0.5
        by = bx * np.tan(theta_rad)
        ax.plot(bx, by, "ks", markersize=12, zorder=3)

        for vec in vectors:
            raw_name = vec.get("name", "")
            name = clean_latex_for_streamlit(raw_name)
            mag = safe_float(vec.get("magnitude", 1.0)) * 0.8
            color = vec.get("color", "red")

            if "N" in raw_name or "Normal" in raw_name:
                dx = -mag * np.sin(theta_rad)
                dy = mag * np.cos(theta_rad)
            elif "f" in raw_name or "Friction" in raw_name:
                dx = -mag * np.cos(theta_rad)
                dy = -mag * np.sin(theta_rad)
            else:
                dx, dy = 0, -mag

            ax.annotate(
                "",
                xy=(bx + dx, by + dy),
                xytext=(bx, by),
                arrowprops=dict(
                    facecolor=color,
                    edgecolor=color,
                    width=1.5,
                    headwidth=6,
                    headlength=8,
                ),
            )
            ax.text(
                bx + dx * 1.25,
                by + dy * 1.25,
                name,
                fontsize=10,
                color=color,
                weight="bold",
                ha="center",
            )

        ax.set_xlim(-1, L + 1)
        ax.set_ylim(-1, L + 1)
        ax.set_aspect("equal")

    elif diag_type == "graph":
        x_data = extra_params.get("x_data", [0, 5])
        y_data = extra_params.get("y_data", [0, 10])
        x_label = extra_params.get("x_label", "Time (s)")
        y_label = extra_params.get("y_label", "Velocity (m/s)")

        ax.plot(x_data, y_data, color="b", linewidth=2.5, marker="o")
        ax.fill_between(x_data, y_data, color="b", alpha=0.15)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        # Polished bounds: tightly frame the data with a clean 10% breathing room
        max_x = max(x_data) if x_data else 5
        max_y = max(y_data) if y_data else 10
        ax.set_xlim(0, max_x * 1.1 if max_x > 0 else 5)
        ax.set_ylim(0, max_y * 1.1 if max_y > 0 else 10)

    else:
        ax.plot(0, 0, "ko", markersize=8, zorder=5)
        if not vectors:
            vectors = []
        max_mag = max(
            [safe_float(v.get("magnitude", 1.0)) for v in vectors], default=1.0
        )
        limit = max(1.8, max_mag * 1.45)

        for vec in vectors:
            raw_name = vec.get("name", "")
            name = clean_latex_for_streamlit(raw_name)
            angle_deg = safe_float(vec.get("angle_deg", 0))
            mag = safe_float(vec.get("magnitude", 1.0))
            color = vec.get("color", "black")

            angle_rad = np.radians(angle_deg)
            dx = mag * np.cos(angle_rad)
            dy = mag * np.sin(angle_rad)

            ax.annotate(
                "",
                xy=(dx, dy),
                xytext=(0, 0),
                arrowprops=dict(
                    facecolor=color,
                    edgecolor=color,
                    width=2,
                    headwidth=7,
                    headlength=9,
                ),
            )
            ax.text(
                dx * 1.18,
                dy * 1.18,
                name,
                fontsize=11,
                color=color,
                ha="center",
                va="center",
                weight="bold",
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=10, pad=10)
    plt.tight_layout()
    return fig


# ==========================================
# 3. OPENAI SINGLE-QUESTION MASTERCLASS GENERATOR
# ==========================================


def generate_single_ap_question(
    course, q_format, q_style, topic, previous_questions, api_key
):
    client = openai.OpenAI(api_key=api_key)

    is_calculus_course = "C:" in course
    if is_calculus_course:
        rigor_guideline = (
            "CALCULUS-BASED RIGOR (AP Physics C): Require calculus applications "
            "such as definite integrals for work/energy or fields, and derivatives for motion."
        )
    else:
        rigor_guideline = (
            "ALGEBRA-BASED RIGOR (AP Physics 1 & 2): Focus on proportional reasoning, "
            "conceptual modeling, and graphical interpretation."
        )

    if q_style == "Conceptual":
        style_instruction = (
            "QUESTION STYLE (Conceptual/Graphical): Center on qualitative reasoning, reading and interpreting graphs "
            "or physical setup models. MANDATORY VISUAL RULE: Set 'has_diagram': true, choose an appropriate "
            "'diag_type' ('fbd', 'incline', or 'graph'). EXACT DATA SYNC & PHYSICS INTEGRITY: If 'diag_type' is 'graph', "
            "you MUST mathematically calculate the exact 'x_data' and 'y_data' array values matching the numbers stated "
            "in the question text with 100% mathematical and physical precision (e.g., quadratic motion curves for distance-time)."
        )
    elif q_style == "Quantitative Arithmetic":
        style_instruction = (
            "QUESTION STYLE (Quantitative Arithmetic): Focus on numerical calculation, algebraic derivations, "
            "and precise quantitative problem-solving."
        )
    elif q_style == "Experimental Design":
        style_instruction = (
            "QUESTION STYLE (Experimental Design): Focus on laboratory setups, data collection, and error analysis. "
            "MANDATORY VISUAL RULE: Set 'has_diagram': true with a relevant physical diagram structure."
        )
    else:
        style_instruction = "Ensure a rigorous balance of conceptual and quantitative application."

    exclusion_context = ""
    if previous_questions:
        exclusion_context = (
            f"EXCLUSION RULE: The teacher has already generated these questions in this test session: {previous_questions}. "
            f"You MUST generate a completely unique, novel scenario testing a different aspect of '{topic}' to ensure variety."
        )

    system_prompt = f"""
    You are an expert AP Physics item writer for College Board exam design in {course}. 
    Generate EXACTLY ONE masterclass-quality, rigorous AP-level question.
    
    {rigor_guideline}
    {style_instruction}
    {exclusion_context}
    
    RIGOROUS PHYSICAL VERIFICATION:
    - Double-check all physics calculations before finalizing options.
    - If a graph is generated, its plotted curve or line must precisely match the physics equations stated in the question text.
    
    PSYCHOMETRIC DISTRACTOR REQUIREMENT:
    Incorrect multiple-choice options (distractors) MUST NOT be random numbers. They must be engineered around 
    well-documented AP student misconceptions (e.g., confusing mass with weight, omitting sine/cosine components, 
    forgetting direction, or mixing up graph slopes/areas).

    Output EXCLUSIVELY in valid JSON format using the EXACT structure below:
    {{
        "scratchpad_derivation": "Step-by-step mathematical solution and physics verification worked out FIRST.",
        "calculated_target_value": "25 m", 
        "question_text": "A car starts from rest and accelerates at a constant rate of 2 m/s^2 for 5 s. What is the total distance traveled?",
        "options": ["A) 10 m", "B) 25 m", "C) 50 m", "D) 5 m"], 
        "correct_answer": "B", 
        "explanation": "Detailed step-by-step derivation matching the calculated_target_value and explaining distractors.",
        "has_diagram": true,
        "diag_type": "graph",
        "vectors": [],
        "extra_params": {{
            "x_data": [0, 5],
            "y_data": [0, 10],
            "x_label": "Time (s)",
            "y_label": "Velocity (m/s)"
        }}
    }}

    CRITICAL RULES:
    1. Solve in 'scratchpad_derivation' FIRST.
    2. 'calculated_target_value' MUST match one of the choices in 'options'.
    3. Format ALL math with single dollar signs ($...$). Never use brackets like \\[ \\].
    4. GRAPHICAL ACCURACY: If diag_type is 'graph', extra_params['x_data'] and extra_params['y_data'] MUST accurately reflect the exact numbers in the question text.
    """

    user_prompt = f"""
    Course: {course}
    Question Format: {q_format}
    Question Style / Focus: {q_style}
    Physics Topic: {topic}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2000,
        temperature=0.5,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_json_str = response.choices[0].message.content
    return sanitize_json_response(raw_json_str)


# ==========================================
# 4. STREAMLIT UI SETUP (SINGLE UNIFIED APP)
# ==========================================

st.set_page_config(
    page_title="AP Physics Exam Builder", page_icon="⚡", layout="centered"
)
st.title("⚡ AP Physics Test Builder & Item Generator")

st.markdown(
    """
*Build your custom AP exam **one rigorous question at a time**. Each generated item is uniquely crafted, 
pedagogically verified, and paired with custom-rendered visual models designed for professional testing.*
"""
)

st.subheader("Exam & Target Settings")

col1, col2 = st.columns(2)

with col1:
    course = st.selectbox(
        "AP Course",
        [
            "AP Physics 1",
            "AP Physics 2",
            "AP Physics C: Mechanics",
            "AP Physics C: Electricity & Magnetism",
        ],
    )

    q_format = st.selectbox(
        "Question Format",
        [
            "Single-Select Multiple Choice (MC)",
            "Multi-Select Multiple Choice (MS)",
            "Free Response Question (FRQ)",
        ],
    )

with col2:
    q_style = st.selectbox(
        "Question Style / Focus",
        [
            "Conceptual",
            "Quantitative Arithmetic",
            "Experimental Design",
            "Random (Mix of All Types)",
        ],
    )

    ap_topics_map = {
        "AP Physics 1": [
            "Kinematics",
            "Dynamics",
            "Circular Motion and Gravitation",
            "Work, Energy, and Power",
            "Linear Momentum",
            "Torque and Rotational Motion",
            "Simple Harmonic Motion",
            "Fluids",
        ],
        "AP Physics 2": [
            "Fluids",
            "Thermodynamics",
            "Electric Force, Field, and Potential",
            "Electric Circuits",
            "Magnetism and Electromagnetism",
            "Optics",
            "Modern Physics",
        ],
        "AP Physics C: Mechanics": [
            "Kinematics",
            "Newton's Laws of Motion",
            "Work, Energy, and Power",
            "Systems of Particles and Linear Momentum",
            "Rotation and Angular Momentum",
            "Oscillations",
            "Gravitation",
        ],
        "AP Physics C: Electricity & Magnetism": [
            "Electrostatics",
            "Conductors, Capacitors, and Dielectrics",
            "Electric Circuits",
            "Magnetic Fields",
            "Electromagnetism",
        ],
    }

    available_topics = ap_topics_map.get(course, ["General Physics"])
    topic = st.selectbox("Physics Topic", available_topics)

# Initialize session state for building a test bank of questions
if "test_bank" not in st.session_state:
    st.session_state["test_bank"] = []

generate_clicked = st.button("Generate Next Test Question", type="primary")

if generate_clicked:
    with st.spinner("Synthesizing rigorous AP item and verifying visual model..."):
        try:
            # Collect previously generated question summaries to avoid duplication
            prev_summaries = [
                q.get("question_text", "")[:50]
                for q in st.session_state["test_bank"]
            ]
            new_q = generate_single_ap_question(
                course,
                q_format,
                q_style,
                topic,
                prev_summaries,
                st.secrets["OPENAI_API_KEY"],
            )
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
    st.markdown(
        f"*Total Questions in Test Bank: {len(st.session_state['test_bank'])}*"
    )

    for i, q in enumerate(st.session_state["test_bank"], start=1):
        st.markdown(f"---")
        q_clean = clean_latex_for_streamlit(q.get("question_text", ""))
        st.markdown(f"### Question {i}\n{q_clean}")

        # Render Polished, Synchronized Diagram / Model
        if q.get("has_diagram", False):
            diag_type = q.get("diag_type", "fbd")
            vectors = q.get("vectors", [])
            extra_params = q.get("extra_params", {})
            fig = draw_synchronized_diagram(
                diag_type,
                vectors,
                title=f"Q{i} Visual Model: {topic}",
                extra_params=extra_params,
            )
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
                    correct_letters = [
                        c.strip()
                        for c in re.split(r"[,&]", raw_correct)
                        if c.strip() in "ABCD"
                    ]

                    if sorted(user_selections) == sorted(correct_letters):
                        st.success(
                            f"🎉 Q{i} Correct! Answers: {', '.join(correct_letters)}."
                        )
                    else:
                        st.error(
                            f"❌ Q{i} Incorrect. Correct answers: {', '.join(correct_letters)}."
                        )
            else:
                user_choice = st.radio(
                    "Options", options, key=f"q_{i}_radio", index=0
                )

                if st.button(f"Check Answer (Q{i})", key=f"check_btn_{i}"):
                    correct_letter = str(q.get("correct_answer", "")).strip()
                    if user_choice.startswith(correct_letter):
                        st.success(
                            f"🎉 Q{i} Correct! Answer: {correct_letter}."
                        )
                    else:
                        st.error(
                            f"❌ Q{i} Incorrect. Correct answer: {correct_letter}."
                        )

        with st.expander(f"View Solution & Explanation — Question {i}"):
            st.markdown(
                f"**Correct Answer:** {q.get('correct_answer', 'N/A')}"
            )
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
                "Comment": comment,
            }

            new_row_df = pd.DataFrame([new_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)

            conn.update(spreadsheet=spreadsheet_url, data=updated_df)
            st.success("Test feedback successfully saved to Google Sheets!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
