import streamlit as st
import pandas as pd
import joblib


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Court to Contract",
    page_icon="🏀",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #e8eef5 100%);
    }

    /* Header */
    .hero {
        background: linear-gradient(135deg, #101820, #1d3557);
        padding: 35px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
    }

    .hero h1 {
        font-size: 48px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 20px;
        color: #dce6f2;
    }

    /* Cards */
    .card {
        background: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0px 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* Section headers */
    .section-title {
        color: #1d3557;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 15px;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #1d3557, #457b9d);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
    }

    .prediction-box h2 {
        font-size: 24px;
    }

    .prediction-box h1 {
        font-size: 42px;
        margin: 10px 0;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background-color: #e76f51;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 10px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #d95d39;
        color: white;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("nba_salary_model.pkl")
feature_names = joblib.load("feature_names.pkl")


# =====================================================
# HERO HEADER
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>🏀 COURT TO CONTRACT</h1>
        <p>NBA Player Salary Prediction Powered by Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# INTRODUCTION
# =====================================================

st.markdown(
    """
    <div class="card">
        <div class="section-title">📊 About the Application</div>
        <p>
        Court to Contract is a machine learning application designed to
        predict NBA player salary performance. The application uses a
        Random Forest Regression model trained on player-related data.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================
# CREATE TWO COLUMNS
# =====================================================

left_column, right_column = st.columns([1.5, 1])


# =====================================================
# PLAYER INPUTS
# =====================================================

with left_column:

    st.markdown(
        """
        <div class="card">
        <div class="section-title">🏀 Player Information</div>
        """,
        unsafe_allow_html=True
    )

    input_data = {}

    for feature in feature_names:

        label = feature.replace("_", " ").title()

        if feature == "salary_cap_year":

            input_data[feature] = st.number_input(
                label,
                min_value=1.0,
                value=100.0
            )

        elif feature in ["age", "years_experience", "experience"]:

            input_data[feature] = st.number_input(
                label,
                min_value=0.0,
                value=25.0
            )

        else:

            input_data[feature] = st.number_input(
                label,
                value=0.0
            )

    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================
# MODEL INFORMATION
# =====================================================

with right_column:

    st.markdown(
        """
        <div class="card">
        <div class="section-title">🤖 Model Information</div>

        <p><b>Algorithm:</b> Random Forest Regression</p>
        <p><b>Target:</b> Salary Cap Percentage</p>
        <p><b>Purpose:</b> NBA Salary Prediction</p>
        <p><b>Model Focus:</b> Generalization and Overfitting Control</p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
        <div class="section-title">📈 How It Works</div>

        <p>1️⃣ Enter player information</p>
        <p>2️⃣ The model analyses the data</p>
        <p>3️⃣ Salary cap percentage is predicted</p>
        <p>4️⃣ The prediction is converted into an estimated salary</p>

        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================
# PREPARE INPUT DATA
# =====================================================

input_df = pd.DataFrame([input_data])

input_df = input_df[feature_names]


# =====================================================
# PREDICTION BUTTON
# =====================================================

st.markdown("---")

if st.button("🏀 PREDICT PLAYER SALARY"):

    predicted_salary_cap_pct = model.predict(input_df)[0]

    st.balloons()

    st.markdown(
        """
        <div class="prediction-box">
        <h2>🏆 Prediction Result</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <h1>{predicted_salary_cap_pct:.2%}</h1>
        <p>Predicted Salary Cap Percentage</p>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # Salary conversion
    if "salary_cap_year" in input_data:

        salary_cap = input_data["salary_cap_year"]

        predicted_salary = (
            predicted_salary_cap_pct * salary_cap
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Salary Cap %",
                f"{predicted_salary_cap_pct:.2%}"
            )

        with col2:
            st.metric(
                "Estimated Salary",
                f"${predicted_salary:,.2f}"
            )

        with col3:
            st.metric(
                "Model",
                "Random Forest"
            )

    else:

        st.info(
            "Salary cap year was not included as a model feature, "
            "so an actual salary estimate could not be calculated."
        )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center; color:#777;'>
    🏀 Court to Contract | NBA Salary Prediction Model<br>
    Developed using Machine Learning and Random Forest Regression
    </div>
    """,
    unsafe_allow_html=True
)