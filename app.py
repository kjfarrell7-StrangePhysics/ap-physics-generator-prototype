import json
import random
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
# 2. HARDENED & DIVERSIFIED DIAGRAM ENGINE
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
        x_data = extra_params.get("x_data", [0, 1, 3, 4])
        y_data = extra_params.get("y_data", [0, 4, 4, 4])
        x_label = extra_params.get("x_label", "Time (s)")
        y_label = extra_params.get("y_label", "Velocity (m/s)")
        shade_area = extra_params.get("shade_area", True)
        curve_type = extra_params.get("curve_type", "linear")
        
        if curve_type == "multi_line":
            y_data_2 = extra_params.get("y_data_2", [0, 2, 4, 6])
            label_1 = extra_params.get("label_1", "Object A")
            label_2 = extra_params.get("label_2", "Object B")
            ax.plot(x_data, y_data, color="#0033cc", linewidth=2.5, marker="o", label=label_1)
            ax.plot(x_data, y_data_2, color="#cc0000", linewidth=2.5, marker="s", label=label_2)
            ax.legend(loc="upper left")
        else:
            ax.plot(x_data, y_data, color="#0033cc", linewidth=3, marker="o")
            if shade_area:
                ax.fill_between(x_data, y_data, color="#0033cc", alpha=0.2)

        ax.set_xlabel(clean_latex_for_streamlit(x_label), fontsize=11, weight="bold")
        ax.set_ylabel(clean_latex_for_streamlit(y_label), fontsize=11, weight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        max_x = max(x_data) if x_data else 5
        all_y = y_data + extra_params.get("y_data_2", []) if curve_type == "multi_line" else y_data
        max_y = max(all_y) if all_y else 10
        min_y = min(all_y) if all_y else 0
        
        ax.set_xlim(0, max_x * 1.05 if max_x > 0 else 5)
        ax.set_ylim(min_y * 1.1 if min_y < 0 else 0, max_y * 1.15 if max_y > 0 else 10)

    elif diag_type == "motion_map":
        positions = extra_params.get("positions", [0, 1, 3, 6, 10])
        ax.axhline(0, color="black", linewidth=1.5, linestyle="-")
        for idx, pos in enumerate(positions):
            ax.plot(pos, 0, marker="o", markersize=10, color="#0033cc")
            ax.text(pos, 0.15, f"$t={idx}\\text{{s}}$", ha="center", fontsize=9, weight="bold")
            if idx > 0:
                prev_pos = positions[idx - 1]
                ax.annotate("", xy=(pos, 0), xytext=(prev_pos, 0),
                            arrowprops=dict(arrowstyle="->", color="#cc0000", lw=1.5, mutation_scale=15))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Position ($m$)", fontsize=11, weight="bold")

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
# 3. DIVERSIFIED AP-READY ITEM GENERATOR
# ==========================================

def generate_single_ap_question(course, q_format, q_style, topic, previous_questions, api_key):
    client = openai.OpenAI(api_key=api_key)

    is_calculus = "C:" in course
    
    if is_calculus:
        rigor = (
            "CALCULUS-BASED AP PHYSICS C MANDATE:\n"
            "- You MUST incorporate proper calculus concepts (derivatives and/or integrals).\n"
            "- For kinematics, use non-constant functions (e.g., $a(t) = bt + c$) requiring explicit integration or differentiation.\n"
            "- Strictly match authentic College Board AP Physics C exam standards."
        )
    else:
        rigor = "ALGEBRA-BASED AP RIGOR: Use standard algebra, trigonometry, vector components, and conceptual physics relationships matching official AP Physics 1 & 2 standards."

    # EXPANDED KINEMATICS GRAPH & VISUAL DIVERSITY ENGINE
    if topic == "Kinematics" and q_style == "Conceptual / Qualitative" and not is_calculus:
        archetype = random.choice([
            "vt_area_displacement", 
            "xt_slope_velocity", 
            "vt_slope_acceleration", 
            "vt_negative_displacement", 
            "at_area_velocity_change", 
            "motion_map_analysis",
            "dual_graph_comparison"
        ])
        
        if archetype == "vt_area_displacement":
            t_ramp = random.choice([1, 2])
            t_total = random.choice([4, 5])
            v_plateau = random.choice([3, 4, 5])
            x_arr = [0, t_ramp, t_total]
            y_arr = [0, v_plateau, v_plateau]
            exact_val = float(0.5 * t_ramp * v_plateau + (t_total - t_ramp) * v_plateau)
            target_str = f"{exact_val:g} m"
            
            unique_options = [target_str, f"{round(exact_val * 0.5, 1):g} m", f"{round(v_plateau * t_total, 1):g} m", f"{round(exact_val + 2.0, 1):g} m"]
            random.shuffle(unique_options)
            correct_letter = ["A", "B", "C", "D"][unique_options.index(target_str)]
            formatted_options = [f"{['A', 'B', 'C', 'D'][idx]}) ${opt}$" if "\\" in opt else f"{['A', 'B', 'C', 'D'][idx]}) {opt}" for idx, opt in enumerate(unique_options)]

            prompt_instruction = f"Write a velocity-time graph question for {course} testing total displacement by calculating the area under the curve. Velocity ramps to {v_plateau} m/s in {t_ramp} s, total time {t_total} s. Verified displacement: {exact_val} m."
            diag_config = {
                "has_diagram": True, "diag_type": "graph",
                "extra_params": {"x_data": x_arr, "y_data": y_arr, "x_label": "Time (s)", "y_label": "Velocity (m/s)", "curve_type": "linear", "shade_area": True}
            }
            target_val_str = target_str

        elif archetype == "xt_slope_velocity":
            t_vals = [0, 2, 4, 6]
            x_vals = [0, 6, 18, 24]
            exact_val = 6.0
            target_str = "6 m/s"
            
            unique_options = ["6 m/s", "3 m/s", "12 m/s", "4 m/s"]
            random.shuffle(unique_options)
            correct_letter = ["A", "B", "C", "D"][unique_options.index(target_str)]
            formatted_options = [f"{['A', 'B', 'C', 'D'][idx]}) ${opt}$" if "\\" in opt else f"{['A', 'B', 'C', 'D'][idx]}) {opt}" for idx, opt in enumerate(unique_options)]

            prompt_instruction = f"Write a position-time graph question for {course} asking the user to find the instantaneous velocity between $t = 2\\text{ s}$ and $t = 4\\text{ s}$ by computing the slope of the curve. Verified velocity: 6 m/s."
            diag_config = {
                "has_diagram": True, "diag_type": "graph",
                "extra_params": {"x_data": t_vals, "y_data": x_vals, "x_label": "Time (s)", "y_label": "Position (m)", "curve_type": "linear", "shade_area": False}
            }
            target_val_str = target_str

        elif archetype == "vt_slope_acceleration":
            t_vals = [0, 3, 5]
            v_vals = [0, 9, 9]
            exact_val = 3.0
            target_str = "3 m/s^2"
            
            unique_options = ["3 m/s^2", "1.8 m/s^2", "9 m/s^2", "4.5 m/s^2"]
            random.shuffle(unique_options)
            correct_letter = ["A", "B", "C", "D"][unique_options.index(target_str)]
            formatted_options = [f"{['A', 'B', 'C', 'D'][idx]}) ${opt}$" if "\\" in opt else f"{['A', 'B', 'C', 'D'][idx]}) {opt}" for idx, opt in enumerate(unique_options)]

            prompt_instruction = f"Write a velocity-time graph question for {course} requiring the student to find the object's acceleration during the linear ramp phase ($t = 0$ to $t = 3\\text{ s}$) using slope. Verified acceleration: 3 m/s^2."
            diag_config = {
                "has_diagram": True, "diag_type": "graph",
                "extra_params": {"x_data": t_vals, "y_data": v_vals, "x_label": "Time (s)", "y_label": "Velocity (m/s)", "curve_type": "linear", "shade_area": False}
            }
            target_val_str = target_str

        elif archetype == "vt_negative_displacement":
            t_vals = [0, 2, 4]
            v_vals = [4, 0, -4] # Triangle above (area = 4) and triangle below (area = -4) -> net displacement = 0
            target_str = "0 m"
            
            unique_options = ["0 m", "8 m", "-8 m", "4 m"]
            random.shuffle(unique_options)
            correct_letter = ["A", "B", "C", "D"][unique_options.index(target_str)]
            formatted_options = [f"{['A', 'B', 'C', 'D'][idx]}) ${opt}$" if "\\" in opt else f"{['A', 'B', 'C', 'D'][idx]}) {opt}" for idx, opt in enumerate(unique_options)]

            prompt_instruction = f"Write a velocity-time graph question for {course} featuring a graph crossing the time axis into negative velocities (linear deceleration from 4 m/s to -4 m/s over 4 s). Ask for the net displacement, testing area above minus area below."
            diag_config = {
                "has_diagram": True, "diag_type": "graph",
                "extra_params": {"x_data": t_vals, "y_data": v_vals, "x_label": "Time (s)", "y_label": "Velocity (m/s)", "curve_type": "linear", "shade_area": True}
            }
            target_val_str = target_str

        elif archetype == "at_area_velocity_change":
            t_vals = [0, 2, 4]
            a_vals = [3, 3, 3] # Constant acceleration = 3 m/s^2 over 4s -> delta v = 12 m/s
            target_str = "12 m/s"
            
            unique_options = ["12 m/s", "6 m/s", "24 m/s", "3 m/s"]
            random.shuffle(unique_options)
            correct_letter = ["A", "B", "C", "D"][unique_options.index(target_str)]
            formatted_options = [f"{['A', 'B', 'C', 'D'][idx]}) ${opt}$" if "\\" in opt else f"{['A', 'B', 'C', 'D'][idx]}) {opt}" for idx, opt in enumerate(unique_options)]

            prompt_instruction = f"Write an acceleration-time graph question for {course} where acceleration is constant at 3 m/s^2 for 4 seconds, asking the student to determine the total change in velocity ($\Delta v$) by calculating the area under the acceleration-time curve."
            diag_config = {
                "has_diagram": True, "diag_type": "graph",
                "extra_params": {"x_data": t_vals, "y_data": a_vals, "x_label": "Time (s)", "y_label": "Acceleration (m/s^2)", "curve_type": "linear", "shade_area": True}
            }
            target_val_str = target_str

        elif archetype == "motion_map_analysis":
            positions = [0, 2, 6, 12, 20] # Increasing spacing -> speeding up
            target_str = "The object is speeding up because the distance between consecutive dots increases with time."
            
            unique_options = [
                "The object is speeding up because the distance between consecutive dots increases with time.",
                "The object is moving at a constant speed because the time intervals are equal.",
                "The object is slowing down because the position values are positive.",
                "The object has zero acceleration throughout the entire motion."
            ]
            correct_letter = "A"
            formatted_options = [f"A) {unique_options[0]}", f"B) {unique_options[1]}", f"C) {unique_options[2]}", f"D) {unique_options[3]}"]

            prompt_instruction = f"Write a motion map analysis question for {course} displaying positions at 1-second intervals with expanding spacing, asking students to interpret velocity and acceleration characteristics."
            diag_config = {
                "has_diagram": True, "diag_type": "motion_map",
                "extra_params": {"positions": positions}
            }
            target_val_str = target_str

        else: # dual_graph_comparison
            t_vals = [0, 2, 4]
            r1_vals = [0, 4, 8]
            r2_vals = [0, 6, 12]
            target_str = "Object B has a higher speed because its position-time slope is steeper."
            
            unique_options = [
                "Object B has a higher speed because its position-time slope is steeper.",
                "Object A has a higher speed because it starts at the origin.",
                "Both objects maintain identical constant velocities.",
                "Object A has a greater acceleration than Object B."
            ]
            correct_letter = "A"
            formatted_options = [f"A) {unique_options[0]}", f"B) {unique_options[1]}", f"C) {unique_options[2]}", f"D) {unique_options[3]}"]

            prompt_instruction = f"Write a dual-graph comparison question for {course} comparing two linear position-time curves with different slopes, asking students to contrast their speeds."
            diag_config = {
                "has_diagram": True, "diag_type": "graph",
                "extra_params": {"x_data": t_vals, "y_data": r1_vals, "y_data_2": r2_vals, "label_1": "Object A", "label_2": "Object B", "x_label": "Time (s)", "y_label": "Position (m)", "curve_type": "multi_line", "shade_area": False}
            }
            target_val_str = target_str
    else:
        correct_letter = "A"
        formatted_options = ["A) Standard Option 1", "B) Standard Option 2", "C) Standard Option 3", "D) Standard Option 4"]
        prompt_instruction = f"Write an authentic, high-rigor {q_format} question for {course}, covering '{topic}' with a style of '{q_style}'. Ensure distractors represent genuine student misconceptions."
        diag_config = {"has_diagram": False, "diag_type": "none", "extra_params": {}}
        target_val_str = "N/A"

    system_prompt = f"""
    You are an expert AP Physics curriculum writer and exam item creator for {course}. 
    {rigor}
    {prompt_instruction}

    Output EXCLUSIVELY in valid JSON format using the EXACT structure below:
    {{
        "scratchpad_derivation": "Show step-by-step mathematical or conceptual physics verification here.",
        "calculated_target_value": "{target_val_str}", 
        "question_text": "Rigorously crafted unique question text using single $...$ for all math variables.",
        "options": {json.dumps(formatted_options)}, 
        "correct_answer": "{correct_letter}", 
        "explanation": "Detailed step-by-step College Board-style explanation showing why the correct choice is valid and why alternate choices fail due to common misconceptions.",
        "has_diagram": {str(diag_config['has_diagram']).lower()},
        "diag_type": "{diag_config['diag_type']}",
        "vectors": [],
        "extra_params": {json.dumps(diag_config['extra_params'])}
    }}
    CRITICAL RULES:
    1. Math must use single $ signs. Never use brackets like \\[ \\].
    2. Match the exact curriculum and high-level rigor of {course}.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=2000,
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate item for Course: {course} | Topic: {topic} | Format: {q_format} | Style: {q_style}"}
        ],
    )
    return sanitize_json_response(response.choices[0].message.content)

# ==========================================
# 4. STREAMLIT UI SETUP
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
    with st.spinner("Synthesizing authentic AP-ready item with diversified visual models..."):
        try:
            prev_questions_context = [
                {
                    "question_text": q.get("question_text", ""),
                    "calculated_target_value": q.get("calculated_target_value", ""),
                    "diag_type": q.get("diag_type", "none")
                } 
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
                # index=None makes the radio button start with no selection (unanswered)
                user_choice = st.radio("Options", options, key=f"q_{i}_radio", index=None)
                
                if st.button(f"Check Answer (Q{i})", key=f"check_btn_{i}"):
                    correct_letter = str(q.get("correct_answer", "")).strip()
                    if user_choice is None:
                        st.warning(f"⚠️ Please select an answer before checking for Q{i}.")
                    elif user_choice.startswith(correct_letter):
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
    rating = st.radio("Rating", ["👍 Good", "👎 Needs Improvement"], key="exam_rating", index=None)
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
                "Rating": rating if rating is not None else "Unrated",
                "Comment": comment
            }
            
            new_row_df = pd.DataFrame([new_row])
            updated_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            
            conn.update(spreadsheet=spreadsheet_url, data=updated_df)
            st.success("Test feedback successfully saved to Google Sheets!")
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
