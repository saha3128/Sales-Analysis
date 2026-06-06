import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Prediction",
    page_icon="💰",
    layout="wide"
)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(filename):
    try:
        with open(f'models/{filename}.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: models/{filename}.pkl")
        st.stop()

try:
    best_model = load_model('best_model')
    scaler = load_model('scaler')
    le_category = load_model('le_category')
    le_product = load_model('le_product')
    le_city = load_model('le_city')
    feature_cols = load_model('feature_cols')
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.info("📌 Please run the notebook's model-saving cell (cell 21) first to create the pickle files.")
    st.stop()

# ── Feature engineering (must match training notebook) ────────────────────────
def engineer_features(df):
    df['Category_Encoded'] = le_category.transform(df['Product_Category'])
    df['Product_Encoded'] = le_product.transform(df['Product_Name'])
    df['City_Encoded'] = le_city.transform(df['Customer_City'])
    
    df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
    df['Price_Per_Quantity'] = df['Unit_Price'] / df['Quantity']
    
    # Using average values from training data
    df['Revenue_Per_City'] = 1125.0
    df['Revenue_Per_Category'] = 1375.0
    df['Revenue_Per_Product'] = 1500.0
    
    return df

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💰 Sales Prediction Model")
st.markdown("Predict total sales for orders based on product and customer details.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    quantity = st.number_input("Quantity", min_value=1, value=1)
    unit_price = st.number_input("Unit Price ($)", min_value=1.0, value=100.0)
    product_category = st.selectbox("Product Category", le_category.classes_)

with col2:
    product_name = st.selectbox("Product Name", le_product.classes_)
    customer_city = st.selectbox("Customer City", le_city.classes_)
    day = st.slider("Day of Month", 1, 31, 15)

with col3:
    day_names = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, 
                "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_of_week = st.selectbox("Day of Week", list(day_names.keys()))
    day_of_week = day_names[day_of_week]

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Sales", use_container_width=True):
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Quantity': [quantity],
        'Unit_Price': [unit_price],
        'Product_Category': [product_category],
        'Product_Name': [product_name],
        'Customer_City': [customer_city],
        'Day': [day],
        'DayOfWeek': [day_of_week],
    })
    
    # Feature engineering
    input_data = engineer_features(input_data)
    
    # Select features
    X = input_data[feature_cols]
    
    # Make prediction
    prediction = best_model.predict(X)[0]
    
    # Display results
    st.success(f"### 💵 Predicted Total Sales: ${prediction:.2f}")
    
    # Show input summary
    st.write("**Order Summary:**")
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.metric("Quantity", quantity)
        st.metric("Unit Price", f"${unit_price:.2f}")
        st.metric("Product", product_name)
    
    with summary_col2:
        st.metric("Category", product_category)
        st.metric("City", customer_city)
        st.metric("Day", f"{day} (Day {day_of_week})")

st.divider()
st.info("ℹ️ This model uses Random Forest to predict total sales based on product and customer features.")

with col1:
    st.subheader("👤 Customer Profile")
    age    = st.number_input("Age",             min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender",             ["Male", "Female"])
    tenure = st.number_input("Tenure (Months)", min_value=0,  max_value=120, value=12)

with col2:
    st.subheader("📊 Behaviour Metrics")
    recency   = st.number_input("Recency (Days since last purchase)", min_value=0,   max_value=730,      value=90)
    frequency = st.number_input("Frequency (Number of transactions)", min_value=1,   max_value=500,      value=20)
    monetary  = st.number_input("Monetary Value (Total spend $)",     min_value=0.0, max_value=100000.0, value=500.0, step=10.0)
    support   = st.number_input("Support Calls",                      min_value=0,   max_value=20,       value=2)

st.divider()

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):

    input_dict = {
        "Age":           age,
        "Gender":        gender,
        "Tenure_Months": tenure,
        "Recency_Days":  recency,
        "Frequency":     frequency,
        "Monetary_Value": monetary,
        "Support_Calls": support,
    }
    df_input = pd.DataFrame([input_dict])
    df_input = engineer_features(df_input)

    proba      = pipe.predict_proba(df_input)[:, 1][0]
    will_churn = proba >= threshold

    st.divider()
    st.subheader("📋 Prediction Result")

    col_res1, col_res2, col_res3 = st.columns(3)

    with col_res1:
        if will_churn:
            st.error("⚠️ **WILL CHURN**")
        else:
            st.success("✅ **RETAINED**")

    with col_res2:
        st.metric("Churn Probability", f"{proba*100:.1f}%")

    with col_res3:
        st.metric("Threshold Used", f"{threshold:.3f}")

    # ── Risk gauge ────────────────────────────────────────────────────────────
    st.markdown("#### Risk Level")
    bar_color  = "🔴" if proba >= 0.7 else ("🟡" if proba >= 0.4 else "🟢")
    risk_label = "High Risk" if proba >= 0.7 else ("Medium Risk" if proba >= 0.4 else "Low Risk")
    st.progress(float(proba), text=f"{bar_color} {risk_label} — {proba*100:.1f}%")

    # ── Risk factors ──────────────────────────────────────────────────────────
    st.markdown("#### ⚡ Key Risk Factors Detected")
    flags = []
    if support >= 7:
        flags.append("🔴 High support calls (≥7) — strong churn signal")
    if recency >= 300:
        flags.append("🔴 High recency (≥300 days) — customer is inactive")
    if tenure <= 6:
        flags.append("🟡 Low tenure (≤6 months) — new customer, higher churn risk")
    if frequency <= 10:
        flags.append("🟡 Low transaction frequency — low engagement")
    if support >= 7 and recency >= 300:
        flags.append("🔴 CRITICAL: High support calls + high recency combined")

    if flags:
        for f in flags:
            st.markdown(f"- {f}")
    else:
        st.markdown("- 🟢 No major risk factors detected")

    # ── Recommendation ────────────────────────────────────────────────────────
    st.markdown("#### 💡 Recommended Action")
    if proba >= 0.7:
        st.warning("**Immediate action needed.** Assign a retention agent, offer a personalised discount or loyalty reward.")
    elif proba >= 0.4:
        st.info("**Monitor closely.** Send a re-engagement email or a small incentive.")
    else:
        st.success("**No action needed.** Customer appears loyal and engaged.")

# ── Batch prediction ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📁 Batch Prediction (CSV Upload)")
st.markdown("Upload a CSV with columns: `Age, Gender, Tenure_Months, Recency_Days, Frequency, Monetary_Value, Support_Calls`")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded is not None:
    batch_df = pd.read_csv(uploaded)
    batch_fe = engineer_features(batch_df.copy())
    probas   = pipe.predict_proba(batch_fe)[:, 1]
    preds    = (probas >= threshold).astype(int)

    batch_df["Churn_Probability"] = (probas * 100).round(1)
    batch_df["Predicted_Churn"]   = preds
    batch_df["Risk"]              = pd.cut(
        probas,
        bins=[0, 0.4, 0.7, 1.0],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )

    st.success(f"✅ Processed {len(batch_df)} customers — "
               f"{preds.sum()} predicted to churn ({preds.mean()*100:.1f}%)")

    col_b1, col_b2, col_b3 = st.columns(3)
    col_b1.metric("Total Customers",   len(batch_df))
    col_b2.metric("Predicted Churners", preds.sum())
    col_b3.metric("Churn Rate",         f"{preds.mean()*100:.1f}%")

    st.dataframe(batch_df.sort_values("Churn_Probability", ascending=False),
                 use_container_width=True)

    csv = batch_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv,
                       file_name="churn_predictions.csv", mime="text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Model: LogisticRegression with SMOTE | Recall: 0.90 | AUC: 0.937 | Threshold: optimised for 90% churn recall")
