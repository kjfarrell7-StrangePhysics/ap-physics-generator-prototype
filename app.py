import os
import pandas as pd
import streamlit as st
from openai import OpenAI
from streamlit_gsheets import GSheetsConnection

# ---------------------------------------------------------
# 1. UTF-8 Safe Environment Setup (Fixes local Windows bugs)
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
        # Connect to Google Sheets via Streamlit connection
        conn = st.connection("gsheets", type=GSheetsConnection)

        # Read existing records
        existing_data = conn.read(ttl=0)

        # Create new feedback record
        new_entry = pd.DataFrame(
            [
                {
                    "Timestamp": pd.Timestamp.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Unit": unit,
                    "Topic": topic,
                    "Question": question_text,
                    "Rating": rating,
                    "Comments": comments,
                }
            ]
        )

        # Append and update sheet
        updated_df = pd.concat([existing_data, new_entry], ignore_index=True)
        conn.update(data=updated_df)
        return True
    except Exception as e:
        # Fallback if Google Sheets secrets are not configured yet
        st.info(f"Note: Feedback captured locally (Google Sheet connection pending).")
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
# 4. Main Application UI
# ---------------------------------------------------------
st.title("⚛️ AP Physics Item Generator")
st.caption(
    "Generate authentic AP-style multiple-choice items with integrated feedback logging."
)

# Topic Selection UI
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
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# ---------------------------------------------------------
# 5. Question Generation Logic
# ---------------------------------------------------------
if st.button("🚀 Generate AP Question", type="primary"):
    st.session_state.feedback_submitted = False  # Reset feedback state for new question

    with st.spinner("Authoring AP-style item and validating math physics..."):
        prompt = f"""
        You are an expert AP Physics test author. Create a original multiple-choice item for:
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

# Display Generated Question
if st.session_state.current_question:
    st.markdown("---")
    st.markdown(st.session_state.current_question)

    # ---------------------------------------------------------
    # 6. Integrated Feedback Loop & Data Collection UI
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Rate & Evolve This Item")
    st.caption(
        "Your feedback helps refine prompt constraints to increase AP exam authenticity worldwide."
    )

    # Streamlit Native Feedback Widget
    rating_index = st.feedback("thumbs", key="item_rating")

    feedback_notes = st.text_area(
        "Notes or Distractor Refinements (Optional)",
        placeholder="e.g., 'Option C needs clearer vector notation' or 'Great conceptual prompt!'",
        key="item_notes",
    )

    if st.button("Submit Feedback", type="secondary"):
        if rating_index is not None:
            rating_label = "Thumbs Up" if rating_index == 1 else "Thumbs Down"

            # Attempt to log feedback to database
            log_feedback_to_sheet(
                unit=unit_option,
                topic=topic_description,
                question_text=st.session_state.current_question,
                rating=rating_label,
                comments=feedback_notes,
            )

            st.session_state.feedback_submitted = True
            st.success("✅ Feedback logged successfully! Thank you for contributing to the dataset.")
        else:
            st.warning("Please click either the 👍 or 👎 icon before submitting.")
