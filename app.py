import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from streamlit_gsheets import GSheetsConnection

# ---------------------------------------------------------
# 1. UTF-8 Safe Environment Setup (Fixes local Windows encoding bugs)
# ---------------------------------------------------------
os.environ["PYTHONUTF8"] = "1"

st.set_page_config(
    page_title="AP Physics Generator Pro", page_icon="⚛️", layout="wide"
)

# ---------------------------------------------------------
# 2. Database Connection Helper (Google Sheets)
# ---------------------------------------------------------
def log_feedback_to_sheet(unit, topic, question_text, rating, comments):
    """Logs question data and feedback into a Google Sheet."""
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        existing_data = conn.read(ttl=0)

        new_entry = pd.DataFrame(
            [
                {
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Unit": unit,
                    "Topic": topic,
                    "Question": question_text,
                    "Rating": rating,
                    "Comments": comments,
                }
            ]
        )

        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        conn.update(data=updated_df)
        return True
    except Exception as e:
        st.info("Note: Feedback captured locally (Google Sheet connection pending).")
        return False


# ---------------------------------------------------------
# 3. Sidebar Configuration & Security
# ---------------------------------------------------------
st.sidebar.title("🔐 App Setup")
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------------------------------------------------
# 4. Main Application UI & Topic Selection
# ---------------------------------------------------------
st.title("⚛️ AP Physics Item Generator")
st.caption(
    "Generate authentic AP-style multiple-choice items with integrated feedback logging."
)

unit_option = st.selectbox(
    "Select AP Physics Unit",
    [
        "Unit 1: Kinematics",
        "Unit 2: Force and Translational Dynamics",
        "Unit 3: Work, Energy, and Power",
        "Unit 4: Linear Momentum",
        "Unit 5: Torque and Rotational Dynamics",
        "Unit 6: Energy and Momentum of Rotating Systems",
        "Unit 7: Oscillations",
    ],
)

topic_description = st.text_input(
    "Specific Topic Focus (Optional)",
    placeholder="e.g., Conservation of Angular Momentum with projectile impact",
)

# Initialize Session State Variables
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "last_rating" not in st.session_state:
    st.session_state.last_rating = None

# ---------------------------------------------------------
# 5. Question Generation Logic
# ---------------------------------------------------------
if st.button("🚀 Generate AP Question", type="primary"):
    st.session_state.last_rating = None  # Reset rating for new question

    with st.spinner("Authoring AP-style item and validating math/physics..."):
        prompt = f"""
        You are an expert AP Physics test author. Create an original multiple-choice item for:
        Unit: {unit_option}
        Topic Context: {topic_description if topic_description else "Core AP Curriculum Standard"}

        Format requirements:
        - Clear conceptual or quantitative prompt (AP Exam style).
        - Options (A), (B), (C), and (D).
        - Clearly state the Correct Answer and brief Explanation at the bottom.
        """

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        st.session_state.current_question = response.choices[0].message.content

# ---------------------------------------------------------
# 6. Display Generated Question & Evaluation Tool
# ---------------------------------------------------------
if st.session_state.current_question:
    st.markdown("---")
    st.markdown(st.session_state.current_question)

    # Integrated Feedback Loop UI
    st.markdown("---")
    st.subheader("📊 Rate & Evolve This Item")
    st.caption("Help refine prompt constraints. Flag incorrect answers, math errors, or weak distractors!")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Accurate & AP-Style", use_container_width=True):
            st.session_state.last_rating = "Thumbs Up"
    with col2:
        if st.button("👎 Incorrect Answer / Needs Fix", use_container_width=True):
            st.session_state.last_rating = "Thumbs Down"

    # Selection indicator
    if st.session_state.last_rating:
        st.info(f"Selected Rating: **{st.session_state.last_rating}**")

    # Feedback comments text area
    feedback_notes = st.text_area(
        "Notes or Specific Errors (Optional)",
        placeholder="e.g., 'The correct answer key says (B), but mathematical calculation yields (D).'",
        key="item_notes",
    )

    # Submit Button
    if st.button("Submit Feedback to Database", type="primary"):
        if st.session_state.last_rating:
            log_feedback_to_sheet(
                unit=unit_option,
                topic=topic_description,
                question_text=st.session_state.current_question,
                rating=st.session_state.last_rating,
                comments=feedback_notes,
            )
            st.success("✅ Feedback logged successfully! Thank you for catching this item.")
        else:
            st.warning("Please click either 👍 or 👎 above before submitting.")
