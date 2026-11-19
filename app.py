
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed"


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# 3. LOAD TEST DATA
# ==========================================

X_test = pd.read_csv(DATA_PATH / "X_test.csv")

y_test = pd.read_csv(
    DATA_PATH / "y_test.csv"
).squeeze("columns")


# ==========================================
# 4. EVALUATE MODEL
# ==========================================

test_predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    test_predictions
)

correct_predictions = (y_test == test_predictions).sum()

total_predictions = len(y_test)

wrong_predictions = total_predictions - correct_predictions

conf_matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=model.classes_
)

report = classification_report(
    y_test,
    test_predictions,
    output_dict=True,
    zero_division=0
)


# ==========================================
# 5. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Prediction System",
    page_icon="🌸",
    layout="wide"
)


# ==========================================
# 6. APPLICATION TITLE
# ==========================================

st.title("🌸 Smart Prediction System")

st.write(
    "A Machine Learning application that predicts "
    "Iris flower species from measurements."
)

st.divider()


# ==========================================
# 7. MODEL PERFORMANCE DASHBOARD
# ==========================================

st.header("📊 Model Performance Dashboard")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

col2.metric(
    "Test Records",
    total_predictions
)

col3.metric(
    "Correct Predictions",
    int(correct_predictions)
)

col4.metric(
    "Incorrect Predictions",
    int(wrong_predictions)
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

st.subheader("Confusion Matrix")

confusion_df = pd.DataFrame(
    conf_matrix,
    index=model.classes_,
    columns=model.classes_
)

st.write(
    "Rows represent actual species. "
    "Columns represent predicted species."
)

st.dataframe(
    confusion_df,
    use_container_width=True
)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

st.subheader("Classification Report")

report_df = pd.DataFrame(report).transpose()

report_df = report_df.round(2)

st.dataframe(
    report_df,
    use_container_width=True
)


st.divider()


# ==========================================
# 10. FLOWER PREDICTION SECTION
# ==========================================

st.header("🌷 Predict a Flower Species")

st.write(
    "Adjust the flower measurements below "
    "and click the prediction button."
)


col1, col2 = st.columns(2)


with col1:

    sepal_length = st.slider(
        "Sepal Length (cm)",
        4.0, 8.0, 5.1, 0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        2.0, 4.5, 3.5, 0.1
    )


with col2:

    petal_length = st.slider(
        "Petal Length (cm)",
        1.0, 7.0, 1.4, 0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        0.1, 2.5, 0.2, 0.1
    )


# ==========================================
# 11. PREDICTION
# ==========================================

if st.button(
    "🔍 Predict Flower Species",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )


    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]


    st.success(
        f"Predicted Species: {prediction.upper()}"
    )


    st.subheader("Prediction Probabilities")


    probability_df = pd.DataFrame({
        "Species": model.classes_,
        "Probability": probabilities
    })


    for species, probability in zip(
        model.classes_,
        probabilities
    ):

        st.write(
            f"**{species.capitalize()}**"
        )

        st.progress(float(probability))

        st.write(
            f"{probability * 100:.2f}%"
        )


# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "Built with Python | Pandas | Scikit-learn | Streamlit"
)
