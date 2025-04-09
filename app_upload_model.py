
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide")
st.title("🧠 Product Quality Prediction Dashboard")
st.sidebar.header("📂 Upload Model & Dataset")

# Upload model file
uploaded_model = st.sidebar.file_uploader("Upload trained model (.pkl)", type=["pkl"])
uploaded_data = st.sidebar.file_uploader("Upload dataset (with 'quality')", type=["csv"])

if uploaded_model is not None and uploaded_data is not None:
    try:
        model = pickle.load(uploaded_model)
        data = pd.read_csv(uploaded_data)

        X = data.drop(columns=["quality"])
        y = data["quality"]

        input_features = {}
        for feature in model.feature_names_in_:
            input_features[feature] = st.sidebar.number_input(f"{feature}", value=0.0)

        input_df = pd.DataFrame([input_features])
        input_df = input_df[model.feature_names_in_]

        if st.button("🔮 Predict Quality"):
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Quality Class: {prediction}")

            st.subheader("📊 Classification Report")
            y_pred = model.predict(X)
            st.text(classification_report(y, y_pred))

            st.subheader("📉 Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            try:
                import shap
                st.subheader("🔍 SHAP Feature Importance")
                explainer = shap.Explainer(model)
                shap_values = explainer(X)
                shap.summary_plot(shap_values, X, show=False)
                st.pyplot(bbox_inches="tight")
            except Exception as e:
                st.warning("SHAP skipped (version issue):")
                st.text(str(e))

    except Exception as e:
        st.error(f"❌ Failed to load or run model: {e}")
else:
    st.info("⬆️ Please upload both model (.pkl) and dataset (.csv) to continue.")
