import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
import matplotlib.pyplot as plt
import numpy as np

# Page Configuration
st.set_page_config(page_title="AP Physics Question Generator", page_icon="⚛️", layout="wide")

# Password Protection Guard
def check_password():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔒 AP Physics Generator - Private Access")
        st.info("Please enter your password to access the app.")
        password_input = st.text_input("Enter Access Password", type="password")
        if st.button("Login"):
            if password_input == "physics2026": 
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        return False
    return True

if not check_password():
    st.stop()

# --- APP HEADER ---
st.title("⚛️ AP Physics Item Generator")
st.caption("Generate AP-style multiple-choice questions with full curriculum units and visual diagram rendering.")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🔑 API Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Paste your sk-... key here")
    
    st.header("📚 Curriculum Settings")
    exam_level = st.selectbox(
        "Select AP Exam Level",
        [
            "AP Physics 1", 
            "AP Physics 2", 
            "AP Physics C: Mechanics", 
            "AP Physics C: Electricity & Magnetism"
        ]
    )
    
    # Unit Lists for All AP Physics Courses
    if exam_level == "AP Physics 1":
        topics = [
            "Unit 1: Kinematics",
            "Unit 2: Force and Translational Dynamics",
            "Unit 3: Work, Energy, and Power",
            "Unit 4: Linear Momentum",
            "Unit 5: Torque and Rotational Dynamics",
            "Unit 6: Energy and Momentum of Rotating Systems",
            "Unit 7: Oscillations & Simple Harmonic Motion (SHM)",
            "Unit 8: Fluids (Density, Pressure, Buoyancy, & Fluid Dynamics)"
        ]
    elif exam_level == "AP Physics 2":
        topics = [
            "Unit 1: Fluids (Density, Pressure, Buoyancy, & Continuity)",
            "Unit 2: Thermodynamics (PV Diagrams, Heat Engines, & First Law)",
            "Unit 3: Electric Force, Field, & Potential",
            "Unit 4: Electric Circuits (DC Circuits, RC Circuits, & Resistors)",
            "Unit 5: Magnetism & Electromagnetic Induction (Faraday's Law)",
            "Unit 6: Geometric & Physical Optics (Reflection, Refraction, Waves)",
            "Unit 7: Quantum, Atomic, & Nuclear Physics (Photoelectric Effect, Half-life)"
        ]
    elif exam_level == "AP Physics C: Mechanics":
        topics = [
            "Kinematics (Calculus-Based)",
            "Newton's Laws & Resistive Forces",
            "Work, Energy, & Conservative Forces",
            "System of Particles & Conservation of Momentum",
            "Rotation & Moments of Inertia via Integration",
            "Oscillations & Simple Harmonic Motion (SHM)",
            "Gravitation & Planetary Motion"
        ]
    else:
        topics = [
            "Electrostatics & Gauss's Law",
            "Electric Potential & Capacitance",
            "Electric Circuits & RC Time Constants",
            "Magnetic Fields & Ampere's Law",
            "Electromagnetic Induction & Faraday's Law"
        ]
    
    selected_topic = st.selectbox("Select Unit / Topic", topics)
    
    cognitive_skill = st.selectbox(
        "Target Cognitive Skill",
        [
            "Qualitative-Quantitative Translation (QQT)",
            "Conceptual Analysis & Proportional Reasoning",
            "Experimental Design & Data Interpretation",
            "Mathematical Derivation & Synthesis"
        ]
    )

# Structured Output Schema with Internal Audit
class APQuestion(BaseModel):
    step_by_step_derivation: str = Field(description="Scratchpad field for solving problem step-by-step")
    conservation_check: str = Field(description="AUDIT CHECK: Verify conservation of momentum, energy, or units. Must state PASS.")
    correct_numerical_value: str = Field(description="Exact computed numerical value or algebraic expression")
    scenario: str
    needs_graph: bool
    graph_type: str  # 'shm_position', 'fluid_depth_pressure', 'force_vs_position', 'kinematics_vt', or 'none'
    question_stem: str
    options: list[str] = Field(description="List of 4 options starting with 'A)', 'B)', 'C)', 'D)' in plain text math formatting")
    correct_answer: str = Field(description="Strictly single letter: 'A', 'B', 'C', or 'D'")
    explanation: str
    misconception_map: list[str]

# Graph Generator Function
def render_physics_graph(graph_type):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    t = np.linspace(0, 10, 200)
    
    if graph_type == "shm_position":
        y = 2 * np.cos(1.5 * t)
        ax.plot(t, y, color="#1f77b4", linewidth=2)
        ax.set_title("Position vs. Time (Simple Harmonic Motion)")
        ax.set_xlabel("Time t (s)")
        ax.set_ylabel("Position x (m)")
        ax.grid(True, linestyle="--", alpha=0.6)
    elif graph_type == "fluid_depth_pressure":
        depth = np.linspace(0, 5, 100)
        pressure = 100 + 10 * depth
        ax.plot(depth, pressure, color="#d62728", linewidth=2)
        ax.set_title("Absolute Pressure vs. Depth in Fluid")
        ax.set_xlabel("Depth h (m)")
        ax.set_ylabel("Pressure P (kPa)")
        ax.grid(True, linestyle="--", alpha=0.6)
    elif graph_type == "force_vs_position":
        x = np.linspace(0, 8, 100)
        F = 12 - 1.5 * x
        ax.plot(x, F, color="#2ca02c", linewidth=2)
        ax.set_title("Force vs. Position")
        ax.set_xlabel("Position x (m)")
        ax.set_ylabel("Force F (N)")
        ax.grid(True, linestyle="--", alpha=0.6)
    else:
        # Default Kinematics v-t graph
        v = 5 + 2 * t - 0.3 * t**2
        ax.plot(t, v, color="#ff7f0e", linewidth=2)
        ax.set_title("Velocity vs. Time")
        ax.set_xlabel("Time t (s)")
        ax.set_ylabel("Velocity v (m/s)")
        ax.grid(True, linestyle="--", alpha=0.6)
        
    st.pyplot(fig)

# Helper Function for Answer Validation
def check_answer(user_choice, correct_answer_field, options):
    if not user_choice or not correct_answer_field:
        return False, "N/A"
    
    user_str = user_choice.strip()
    target_raw = correct_answer_field.strip()
    
    target_letter = None
    if len(target_raw) == 1 and target_raw.upper() in ['A', 'B', 'C', 'D']:
        target_letter = target_raw.upper()
    elif target_raw.startswith("Option ") and len(target_raw) >= 8 and target_raw[7].upper() in ['A', 'B', 'C', 'D']:
        target_letter = target_raw[7].upper()
    elif target_raw[0].upper() in ['A', 'B', 'C', 'D']:
        target_letter = target_raw[0].upper()
        
    user_letter = None
    if user_str in options:
        idx = options.index(user_str)
        user_letter = ['A', 'B', 'C', 'D'][idx]
    elif user_str[0].upper() in ['A', 'B', 'C', 'D']:
        user_letter = user_str[0].upper()
        
    is_correct = (user_letter == target_letter) if (user_letter and target_letter) else (user_str == target_raw)
    return is_correct, (target_letter or target_raw)

# --- MAIN PAGE GENERATION ---
if st.button("🚀 Generate AP Question", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API Key in the sidebar to proceed.")
    else:
        try:
            client = OpenAI(api_key=api_key)
            
            system_prompt = f"""
            You are a senior AP Physics test author for College Board.
            Create a unique, creative, and mathematically pristine multiple-choice question for:
            - Exam Level: {exam_level}
            - Topic: {selected_topic}
            - Cognitive Skill: {cognitive_skill}

            STRICT MATHEMATICAL AUDIT & EXECUTION ORDER:
            1. STEP 1 (Derivation): Solve the physics problem step-by-step in 'step_by_step_derivation'.
            2. STEP 2 (Conservation Check): Verify conservation laws (e.g. p_initial == p_final, KE_initial == KE_final for elastic, units match). Record in 'conservation_check'. If math fails, recalculate.
            3. STEP 3 (Correct Value): Store exact result in 'correct_numerical_value'.
            4. STEP 4 (Options): Populate 'options' with plain text math formatting (NO RAW LATEX LIKE '\\frac{{1}}{{2}}'). One option MUST be 'correct_numerical_value'. Distractors must represent real physics misconceptions.
            5. STEP 5 (Answer Key): Set 'correct_answer' to strictly ONE LETTER ONLY ('A', 'B', 'C', or 'D').
            6. GRAPHING: Set 'needs_graph' to true if interpreting a figure/graph, and choose 'graph_type' from 'shm_position', 'fluid_depth_pressure', 'force_vs_position', 'kinematics_vt', or 'none'.
            """

            with st.spinner("Generating AP Physics Question & Running Conservation Audit..."):
                response = client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    temperature=0.7,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Generate a fresh, mathematically verified {exam_level} question on {selected_topic} focusing on {cognitive_skill}."}
                    ],
                    response_format=APQuestion
                )
                
                q_data = response.choices[0].message.parsed
                st.session_state["current_question"] = q_data

        except Exception as e:
            st.error(f"Generation failed: {e}")

# --- DISPLAY QUESTION CARD ---
if "current_question" in st.session_state:
    q = st.session_state["current_question"]
    
    st.markdown("---")
    st.subheader("📝 Practice Question")
    
    st.write(q.scenario)
    
    # Render Actual Visual Graph if needed
    if q.needs_graph and q.graph_type != "none":
        st.markdown("#### 📊 Reference Graph")
        render_physics_graph(q.graph_type)
        
    st.markdown(f"**{q.question_stem}**")
    
    # Multiple Choice Options
    user_choice = st.radio("Select your answer:", q.options, index=None)
    
    if st.button("Submit Answer"):
        if user_choice is None:
            st.warning("Please select an answer option first.")
        else:
            is_correct, correct_letter = check_answer(user_choice, q.correct_answer, q.options)
            
            if is_correct:
                st.success(f"🎉 Correct! Option {correct_letter} is the right answer.")
            else:
                st.error(f"❌ Incorrect. The correct answer is Option {correct_letter}.")
                
    # Instructor Solutions & Misconceptions
    with st.expander("🔍 View Instructor Solutions & Misconception Map"):
        st.markdown("### Correct Answer Explanation")
        st.write(q.explanation)
        
        st.markdown("### Internal Math & Conservation Audit")
        st.info(f"**Verification Check:** {q.conservation_check}")
        
        st.markdown("### Distractor Misconception Analysis")
        for misc in q.misconception_map:
            st.markdown(f"- {misc}")
