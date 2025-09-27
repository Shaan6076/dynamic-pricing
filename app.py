import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load model + features
model = joblib.load("models/dynamic_pricing_model.pkl")
features = joblib.load("models/model_features.pkl")

# Page setup
st.set_page_config(page_title="Dynamic Pricing AI", page_icon="📊", layout="centered")
st.title("🛒 Dynamic Pricing - Sales Prediction")

st.write("This app predicts sales based on product details. Try single product prediction or upload multiple products.")

# ---------------------------
# 🔹 Single Product Prediction
# ---------------------------
st.header("Single Product Prediction")

price = st.slider("Price", 10, 5000, 450)
cost = st.slider("Cost", 10, 4000, 300)
gender = st.selectbox("Gender", ["male", "female"])
category = st.selectbox("Category", ["jeans", "jacket", "shoes", "t-shirt", "top", "trainers"])
brand = st.selectbox("Brand", ["brand_2", "brand_3", "brand_4"])
collection = st.selectbox("Collection", ["P", "SS"])
price_tier = st.selectbox("Price Tier", ["low", "middle"])
style = st.selectbox("Style", ["sport", "casual"])

# Auto-generate date features
today = datetime.today()
year, month, day = today.year, today.month, today.day
dayofweek = today.weekday()
is_weekend = 1 if dayofweek in [5,6] else 0

# Convert input to model-ready format
sample = {
    "price": price,
    "cost": cost,
    "year": year, "month": month, "day": day,
    "dayofweek": dayofweek, "is_weekend": is_weekend,
    "gender_male": 1 if gender=="male" else 0,
    "category_jacket": 1 if category=="jacket" else 0,
    "category_jeans": 1 if category=="jeans" else 0,
    "category_shoes": 1 if category=="shoes" else 0,
    "category_t-shirt": 1 if category=="t-shirt" else 0,
    "category_top": 1 if category=="top" else 0,
    "category_trainers": 1 if category=="trainers" else 0,
    "brand_brand_2": 1 if brand=="brand_2" else 0,
    "brand_brand_3": 1 if brand=="brand_3" else 0,
    "brand_brand_4": 1 if brand=="brand_4" else 0,
    "collection_P": 1 if collection=="P" else 0,
    "collection_SS": 1 if collection=="SS" else 0,
    "price_tier_low": 1 if price_tier=="low" else 0,
    "price_tier_middle": 1 if price_tier=="middle" else 0,
    "style_sport": 1 if style=="sport" else 0,
}

sample_df = pd.DataFrame([sample])[features]

if st.button("Predict Sales"):
    prediction = model.predict(sample_df)[0]
    st.success(f"📊 Predicted Sales: **{prediction:.2f} units**")

# ---------------------------
# 🔹 Multiple Products Prediction
# ---------------------------
st.header("Multiple Products Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV with product details", type=["csv"])
if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    # Reindex to model features
    X_new = new_data.reindex(columns=features, fill_value=0)
    preds = model.predict(X_new)
    new_data["Predicted_Sales"] = preds
    st.write("### Predictions Table")
    st.dataframe(new_data)

    # Plot bar chart
    st.write("### Predicted Sales Bar Chart")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=new_data.index, y=new_data["Predicted_Sales"], palette="Blues_d", ax=ax)
    ax.set_xlabel("Product Index")
    ax.set_ylabel("Predicted Sales")
    st.pyplot(fig)

# ---------------------------
# 🔹 Model Evaluation Visuals
# ---------------------------
st.header("Model Evaluation (Demo)")

chart_type = st.radio(
    "Choose comparison chart:",
    ("None", "Scatter Plot", "Bar Chart (20 Samples)")
)

if chart_type != "None":
    try:
        import joblib
        import matplotlib.pyplot as plt

        y_test = joblib.load("models/y_test.pkl")
        y_pred = joblib.load("models/y_pred.pkl")

        if chart_type == "Scatter Plot":
            fig, ax = plt.subplots(figsize=(7,6))
            ax.scatter(y_test, y_pred, alpha=0.5, color="purple")
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
            ax.set_xlabel("Actual Sales")
            ax.set_ylabel("Predicted Sales")
            ax.set_title("Actual vs Predicted Sales (Scatter)")
            st.pyplot(fig)

        elif chart_type == "Bar Chart (20 Samples)":
            compare_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).sample(20, random_state=42)
            fig, ax = plt.subplots(figsize=(12,6))
            compare_df.reset_index(drop=True).plot(kind="bar", ax=ax)
            ax.set_title("Actual vs Predicted Sales (20 Random Samples)")
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Sales")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Could not load evaluation data: {e}\n\n"
                 "Make sure you saved y_test.pkl and y_pred.pkl from your notebook.")


