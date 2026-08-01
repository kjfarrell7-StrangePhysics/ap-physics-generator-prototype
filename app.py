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

    # Fix literal '\n' text strings leaked by JSON sanitization into actual newlines
    text = text.replace(r"\n", "\n")

    # Clean \boldsymbol or mangled \b backspace escape artifacts -> convert to standard \vec{}
    text = re.sub(r"[\x08]oldsymbol\{([^}]+)\}", r"\\vec{\1}", text)
    text = re.sub(r"\\boldsymbol\{([^}]+)\}", r"\\vec{\1}", text)

    # Convert \[ ... \] to $$ ... $$
    text = re.sub(r"\\\[\s*", "$$", text)
    text = re.sub(r"\s*\\\]", "$$", text)

    # Convert \( ... \) to $ ... $
    text = re.sub(r"\\\(\s*", "$", text)
    text = re.sub(r"\s*\\\)", "$", text)

    return text


def sanitize_json_response(raw_str: str) -> dict:
    """Robustly converts raw single LaTeX backslashes in GPT output (e.g. \\theta, \\frac, \\mu)

    into double backslashes so json.loads() won't crash on invalid JSON escape sequences.
    """

    def fix_slash(match):
        group = match.group(0)
        if group == "\\\\":
            return "\\\\"  # Preserve existing double backslashes
        return "\\\\"  # Convert single backslash to double backslash

    # Match either existing \\ OR any single \ that is NOT followed by " or \
    cleaned = re.sub(r'\\\\|\\(?!["\\])', fix_slash, raw_str)

    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        # Fallback: force escape backslashes across raw payload
        cleaned_fallback = re.sub(r'\\(?!["\\])', r"\\\\", raw_str)
        return json.loads(cleaned_fallback, strict=False)


# ==========================================
# 2. VECTOR DIAGRAM ENGINE
# ==========================================


def safe_float(val, default=1.0):
    """Safely converts model-generated numbers or strings to floats to prevent TypeErrors."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def draw_vector_diagram(vectors, title="Free-Body / Vector Diagram"):
    """Generates a geometrically accurate vector diagram using Matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 4))

    # Center object (point mass)
    ax.plot(0, 0, "ko", markersize=8, zorder=5)

    if not vectors:
        vectors = []

    # Safely extract magnitudes and handle string/numeric mismatches from LLM output
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

        # Convert polar to Cartesian
        angle_rad = np.radians(angle_deg)
        dx = mag * np.cos(angle_rad)
        dy = mag * np.sin(angle_rad)

        # Draw vector arrow
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
                shrink=0,
            ),
        )

        # Offset label slightly past the arrowhead
        lx, ly = dx * 1.18, dy * 1.18
        ax.text(
            lx,
            ly,
            name,
            fontsize=11,
            color=color,
            ha="center",
            va="center",
            weight="bold",
        )

    # Maintain 1:1 aspect ratio so incline angles do not distort
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # Clean axes grid
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(True, linestyle=":", alpha=0.4)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=10, pad=10)

    plt.tight_layout()
    return fig


# ==========================================
# 3. OPENAI BULK EXAM GENERATOR LOGIC
# ==========================================


def generate_ap_exam(course, q_format, q_style, topic, num_questions, api_key):
    client = openai.OpenAI(api_key=api_key)

    # Automatically enforce rigor rules based on course type
    is_calculus_course = "C:" in course
    if is_calculus_course:
        rigor_guideline = (
            "CALCULUS-BASED RIGOR (AP Physics C): The questions must explicitly"
            " demand calculus applications where appropriate (e.g., evaluating"
            " definite integrals for position-dependent forces"
            " W = \\int F(x)dx, integration for rotational inertia"
            " I = \\int r^2 dm, derivatives for velocity/acceleration, or"
            " differential equations for circuit transients)."
        )
    else:
        rigor_guideline = (
            "ALGEBRA-BASED RIGOR (AP Physics 1 & 2): Focus on proportional"
            " reasoning, conceptual modeling, algebraic manipulation, and"
            " graphical interpretation without requiring calculus."
        )

    # Tailor style instructions and strictly enforce visual/graph generation for conceptual items
    if q_style == "Conceptual":
        style_instruction = (
            "QUESTION STYLE (Conceptual/Graphical): Center the item heavily on"
            " qualitative reasoning, reading and interpreting graphs (such as"
            " position-time, velocity-time, potential wells, force-position,"
            " or field lines), analyzing slopes and areas under curves, and"
            " evaluating motion maps. MANDATORY VISUAL RULE: For conceptual and"
            " graphical questions, you MUST set 'has_diagram': true and provide"
            " realistic vector or component definitions inside the 'vectors'"
            " array to accompany the graph description."
        )
    elif q_style == "Quantitative Arithmetic":
        style_instruction = (
            "QUESTION STYLE (Quantitative Arithmetic): Focus on numerical"
            " calculation, algebraic derivations, symbolic variable solving,"
            " and precise quantitative problem-solving with proper dimensional"
            " analysis."
        )
    elif q_style == "Experimental Design":
        style_instruction = (
            "QUESTION STYLE (Experimental Design): Focus on laboratory setups,"
            " identifying experimental uncertainties, designing data collection"
            " procedures, error analysis, and data linearization techniques."
            " MANDATORY VISUAL RULE: Set 'has_diagram': true where a setup or"
            " vector representation clarifies the experiment."
        )
    else:
        style_instruction = (
            "Ensure the set includes a balanced mix of conceptual/graphical,"
            " quantitative arithmetic, and experimental design questions across"
            " the generated set. Enable 'has_diagram': true for conceptual or"
            " vector-heavy items."
        )

    system_prompt = f"""
    You are an expert AP Physics exam writer and College Board item designer for {course}. 
    Generate a complete assessment containing exactly {num_questions} DISTINCT, non-repeating questions based on the user parameters.
    
    {rigor_guideline}
    {style_instruction}
    
    Ensure each question covers a unique angle, scenario, or sub-concept within the topic so there is zero redundancy across the set.

    Output EXCLUSIVELY in valid JSON format using the EXACT structure below:
    {{
        "questions": [
            {{
                "scratchpad_derivation": "Step-by-step mathematical solution worked out FIRST.",
                "calculated_target_value": "4.41 V", 
                "question_text": "The problem description. Use standard single dollar sign LaTeX ($...$) for math formulas.",
                "options": ["A) 7.6 V", "B) 4.4 V", "C) 12.0 V", "D) 0.0 V"], 
                "correct_answer": "B", 
                "explanation": "Detailed step-by-step mathematical derivation matching the calculated_target_value.",
                "has_diagram": true, 
                "vectors": [
                    {{"name": "$F_g$", "angle_deg": 270, "magnitude": 1.0, "color": "blue"}}
                ]
            }}
        ]
    }}

    CRITICAL EXECUTION ORDER & ACCURACY RULES:
    1. ALWAYS solve the problem in 'scratchpad_derivation' FIRST before outputting 'options'.
    2. 'calculated_target_value' MUST be explicitly included as one of the choices in 'options'.
    3. NEVER fabricate or lie about mathematical results in 'explanation' to fit a bad option choice.
    4. Format ALL math variables and equations with single dollar signs ($...$). NEVER use brackets like \\[ \\] or \\( \\).
    5. DO NOT use \\boldsymbol or complex macros for vectors. Always use \\vec{{F}} or standard capital letters like F.
    6. PHYSICS & TERMINOLOGY RULES:
       - NUMERICAL VS. SYMBOLIC CONSISTENCY: If the question stem uses variable symbols without explicit numerical values, ALL options MUST be symbolic expressions.
       - ALGEBRAIC INTEGRITY: When solving symbolic derivations, write out every fraction inversion step explicitly. Never drop numerical coefficients during simplification.
       - CIRCUITS TERMINOLOGY: Do NOT label a circuit as 'LRC' unless an inductor (L) is explicitly present. Use 'RC circuit' or 'RL circuit' when only two components exist. Always specify if a circuit is 'charging' or 'discharging'.
       - RC / RL CIRCUITS NUMERICAL VALUES: Set time 't' to be an exact multiple or standard fraction of the time constant tau (e.g., t = tau, t = 2*tau).
    """

    user_prompt = f"""
    Course: {course}
    Question Format: {q_format}
    Question Style / Focus: {q_style}
    Physics Topic: {topic}
    Number of Questions Required: {num_questions}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=3000,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw_json_str = response.choices[0].message.content
    return sanitize_json_response(raw_json_str)


# ==========================================
# 4. STREAMLIT UI SETUP (NO PASSWORD REQUIRED)
# ==========================================

st.set_page_config(
    page_title="AP Physics Exam Generator", page_icon="⚡", layout="centered"
)
st.title("⚡ AP Physics Exam & Question Generator")

st.subheader("Exam Configuration")

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

    # Course-specific topic dictionary mapped to official AP objectives
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

    # Dynamically filter topics based on the selected course
    available_topics = ap_topics_map.get(course, ["General Physics"])
    topic = st.selectbox("Physics Topic", available_topics)

# Question Count Slider (capped at 5)
num_questions = st.slider(
    "Number of Questions in Exam Set", min_value=1, max_value=5, value=3
)

# Generation Trigger
if st.button("Generate Exam Set", type="primary"):
    with st.spinner(
        f"Drafting custom {num_questions}-question AP exam set..."
    ):
        try:
            data = generate_ap_exam(
                course,
                q_format,
                q_style,
                topic,
                num_questions,
                st.secrets["OPENAI_API_KEY"],
            )
            st.session_state["current_exam"] = data
        except Exception as e:
            st.error(f"Error generating exam: {e}")

# Display Exam Output
if "current_exam" in st.session_state and "questions" in st.session_state["current_exam"]:
    exam_data = st.session_state["current_exam"]
    questions = exam_data.get("questions", [])

    st.divider()
    st.header(f"📋 Generated {course} Assessment ({topic})")
    st.markdown(
        f"*Total Questions: {len(questions)} | Format: {q_format}*"
    )

    # Loop through each question in the bulk exam set
    for i, q in enumerate(questions, start=1):
        st.markdown(f"---")
        q_clean = clean_latex_for_streamlit(q.get("question_text", ""))
        st.markdown(f"### Question {i}\n{q_clean}")

        # Render Vector Diagram if applicable
        if q.get("has_diagram", False) and q.get("vectors", []):
            fig = draw_vector_diagram(
                q.get("vectors", []), title=f"Q{i} Diagram: {topic}"
            )
            st.pyplot(fig)

        options = [
            clean_latex_for_streamlit(opt) for opt in q.get("options", [])
        ]

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
                    correct_letter = str(
                        q.get("correct_answer", "")
                    ).strip()
                    if user_choice.startswith(correct_letter):
                        st.success(
                            f"🎉 Q{i} Correct! Answer: {correct_letter}."
                        )
                    else:
                        st.error(
                            f"❌ Q{i} Incorrect. Correct answer: {correct_letter}."
                        )

        # Solution Expander for each question
        with st.expander(f"View Solution & Explanation — Question {i}"):
            st.markdown(
                f"**Correct Answer:** {q.get('correct_answer', 'N/A')}"
            )
            st.markdown(
                clean_latex_for_streamlit(q.get("explanation", ""))
            )

    # --- EXAM FEEDBACK LOGGING ---
    st.divider()
    st.subheader("Provide Feedback on Exam Set")
    rating = st.radio("Rating", ["👍 Good", "👎 Needs Improvement"], key="exam_rating")
    comment = st.text_area("Comments / Bug Details for this Exam", key="exam_comment")

    if st.button("Submit Exam Feedback", key="submit_exam_feedback"):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            spreadsheet_url = st.secrets["connections"]["gsheets"][
                "spreadsheet"
            ]
            existing_df = conn.read(spreadsheet=spreadsheet_url)

            new_row = {
                "Timestamp": str(np.datetime64("now")),
                "Course": course,
                "Format": q_format,
                "Style": q_style,
                "Topic": topic,
                "QuestionCount": len(questions),
                "Rating": rating,
                "Comment": comment,
            }

            new_row_df = pd.DataFrame([new_row])
            updated_df = pd.concat(
                [existing_df, new_row_df], ignore_index=True
            )

            conn.update(spreadsheet=spreadsheet_url, data=updated_df)
            st.success("Exam feedback successfully saved to Google Sheets!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
