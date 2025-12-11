import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Estimator", layout="centered")
st.title("🏠 House Price Estimator")

st.write("Estimate house prices based on basic property attributes.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
# ----------------------------
# Load model & columns
# ----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("linear_regression_model.pkl")
    cols = joblib.load("model_columns.pkl")
    return model, cols

model, model_columns = load_artifacts()

has_coef = hasattr(model, "coef_")
has_fi = hasattr(model, "feature_importances_")

binary_cols = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(row: dict) -> pd.DataFrame:
    df = pd.DataFrame([row])

    for col in binary_cols:
        df[col] = df[col].map({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

    # align columns
    for c in model_columns:
        if c not in df.columns:
            df[c] = 0

    df = df[model_columns]
    return df


def explain(pre_row: pd.DataFrame, top_k=3):
    if has_coef:
        vals = pre_row.values[0] * model.coef_
    elif has_fi:
        vals = pre_row.values[0] * model.feature_importances_
    else:
        return []

    idx = vals.argsort()[::-1][:top_k]
    return [{"feature": model_columns[i], "impact": round(float(vals[i]), 3)} for i in idx]


# ----------------------------
# Single Prediction
# ----------------------------
with tab1:
    st.subheader("Single Prediction")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input("Area (sqft)", 200, 10000, 1200)
        bedrooms = st.number_input("Bedrooms", 0, 10, 3)
        bathrooms = st.number_input("Bathrooms", 0, 10, 2)
        stories = st.selectbox("Stories", [1, 2, 3, 4])
        parking = st.number_input("Parking", 0, 3, 1)

    with col2:
        mainroad = st.selectbox("Mainroad", ["yes", "no"])
        guestroom = st.selectbox("Guestroom", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
        airconditioning = st.selectbox("Airconditioning", ["yes", "no"])
        prefarea = st.selectbox("Preferred Area", ["yes", "no"])

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

    if st.button("Predict Price"):
        row = {
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "mainroad": mainroad,
            "guestroom": guestroom,
            "basement": basement,
            "hotwaterheating": hotwaterheating,
            "airconditioning": airconditioning,
            "parking": parking,
            "prefarea": prefarea,
            "furnishingstatus": furnishingstatus
        }

        try:
            pre = preprocess(row)
            pred = float(model.predict(pre)[0])
            st.metric("Estimated Price", f"₹ {pred:,.2f}")

            exp = explain(pre)
            if exp:
                st.write("Top contributing features:")
                st.table(pd.DataFrame(exp))

        except Exception as e:
            st.error(f"Error: {e}")


# ----------------------------
# Batch Prediction
# ----------------------------
with tab2:
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview:", df.head())

            preds = []
            for _, row in df.iterrows():
                pre = preprocess(row.to_dict())
                pred = float(model.predict(pre)[0])
                preds.append(pred)

            df["predicted_price"] = preds
            st.write(df.head())

            csv_out = df.to_csv(index=False)
            st.download_button("Download Results", csv_out, "predictions.csv")

        except Exception as e:
            st.error(f"Failed to process file: {e}")
