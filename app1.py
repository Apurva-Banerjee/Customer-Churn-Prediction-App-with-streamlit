import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# ==========================
# Load Model
# ==========================
model = tf.keras.models.load_model("model.h5")   # Use model.h5 if you haven't converted

# ==========================
# Load Encoders & Scaler
# ==========================
with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehotencoder.pkl", "rb") as f:
    onehotencoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Customer Churn Prediction")

st.title("🏦 Customer Churn Prediction")

st.write("Enter customer details below.")

# ==========================
# User Inputs
# ==========================

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650,
)

geography = st.selectbox(
    "Geography",
    onehotencoder.categories_[0]
)

gender = st.selectbox(
    "Gender",
    label_encoder_gender.classes_
)

age = st.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.slider(
    "Tenure",
    0,
    10,
    5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.slider(
    "Number of Products",
    1,
    4,
    1
)

has_card = st.selectbox(
    "Has Credit Card",
    [0, 1]
)

active_member = st.selectbox(
    "Is Active Member",
    [0, 1]
)

salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# ==========================
# Prediction Button
# ==========================

if st.button("Predict Churn"):

    # Label Encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Base DataFrame
    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender_encoded],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_card],
        "IsActiveMember": [active_member],
        "EstimatedSalary": [salary]
    })

    # One-Hot Encode Geography
    geo_encoded = onehotencoder.transform(input_df[["Geography"]])

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehotencoder.get_feature_names_out(["Geography"])
    )

    # Remove Geography column
    input_df = input_df.drop("Geography", axis=1)

    # Merge
    final_input = pd.concat(
        [input_df.reset_index(drop=True),
         geo_encoded_df.reset_index(drop=True)],
        axis=1
    )

    # Scale
    final_input_scaled = scaler.transform(final_input)

    # Predict
    prediction = model.predict(final_input_scaled)

    probability = prediction[0][0]

    st.subheader("Prediction Result")

    st.write(f"**Churn Probability:** {probability:.2%}")

    if probability > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")