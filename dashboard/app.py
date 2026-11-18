
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="X-HIDS | Cyber Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from predictor import XHIDSPredictor

# --------------------------------------------------
# CUSTOM THEME
# --------------------------------------------------



# SESSION STATE
# --------------------------------------------------

if "last_predictions" not in st.session_state:
    st.session_state.last_predictions = None

if "alert_history" not in st.session_state:
    st.session_state.alert_history = []

if "last_filename" not in st.session_state:
    st.session_state.last_filename = None

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

@st.cache_resource
def load_engine():
    return XHIDSPredictor(BASE)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ X-HIDS")
st.sidebar.caption("Explainable Hybrid Intrusion Detection")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Command Center",
        "CSV Detection",
        "Known Attack Detection",
        "Anomaly Detection",
        "Hybrid Risk Analysis",
        "Explainable AI",
        "Model Performance",
        "Alert History"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("""
<div style="
    background-color: #12332f;
    color: #b9f5df;
    border: 1px solid #278b70;
    border-radius: 8px;
    padding: 12px;
    font-weight: 600;
">
    ✓ Research models packaged
</div>
""", unsafe_allow_html=True)
st.sidebar.caption("CIC-IDS2017 | 71 Features | 15 Classes")
st.sidebar.caption("Research prototype — uploaded flow analysis")

# --------------------------------------------------
# COMMAND CENTER
# --------------------------------------------------

if page == "Command Center":

    st.title("X-HIDS Cyber Command Center")

    st.caption(
        "Explainable Hybrid AI-Based Intrusion Detection "
        "for Known and Unseen Attack Categories"
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Input Features", "71")
    c2.metric("Attack Classes", "14")
    c3.metric("Detection Models", "4")
    c4.metric("Alert Threshold", "40 / 100")

    st.subheader("System Architecture")

    st.markdown("""
    **Network Flow CSV**
    
    ↓

    **Preprocessing**
    - Feature validation and alignment
    - Original negative-value cleaning
    - Training-median imputation
    - Standardization for anomaly models

    ↓

    **AI Detection Layer**
    - Random Forest
    - XGBoost
    - Isolation Forest
    - Autoencoder

    ↓

    **Hybrid Risk Engine**
    - Supervised attack probabilities
    - Anomaly severity
    - Combined risk score

    ↓

    **Explainability and Response**
    - SHAP explanations
    - Threat severity
    - Response recommendations
    """)

    st.subheader("Current Session")

    total_alerts = len(st.session_state.alert_history)

    st.metric("Generated Alerts", total_alerts)

    st.info(
        "This is an offline research prototype. "
        "It analyzes uploaded network-flow CSV files and "
        "does not capture live network packets."
    )

# --------------------------------------------------
# CSV DETECTION
# --------------------------------------------------

elif page == "CSV Detection":

    st.title("Network Traffic Detection")

    st.write(
        "Upload a CSV containing all 71 required numerical "
        "network-flow features."
    )

    uploaded = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        key="traffic_csv"
    )

    if uploaded is not None:

        try:
            input_df = pd.read_csv(uploaded)

            st.success("CSV uploaded successfully.")

            st.write("Input records:", len(input_df))
            st.write("Input columns:", len(input_df.columns))

            st.subheader("Uploaded Data Preview")
            st.dataframe(
                input_df.head(10),
                use_container_width=True
            )

            if st.button(
                "Run X-HIDS Detection",
                type="primary"
            ):

                try:
                    with st.spinner(
                        "Loading models and analyzing network traffic..."
                    ):
                        engine = load_engine()
                        predictions = engine.predict(input_df)

                    st.session_state.last_predictions = predictions
                    st.session_state.last_filename = uploaded.name

                    alerts = predictions[
                        predictions["Hybrid_Prediction"]
                    ].copy()

                    if len(alerts):
                        history = alerts.copy()
                        history.insert(
                            0, "Source_File", uploaded.name
                        )

                        st.session_state.alert_history.extend(
                            history.to_dict("records")
                        )

                    st.success(
                        f"Analysis completed: {len(predictions)} records processed."
                    )

                except Exception as e:
                    st.error(f"Detection failed: {e}")

            if st.session_state.last_predictions is not None:

                predictions = st.session_state.last_predictions

                st.divider()
                st.subheader("Detection Summary")

                total = len(predictions)
                alerts = int(
                    predictions["Hybrid_Prediction"].sum()
                )
                benign = total - alerts

                a, b, c, d = st.columns(4)

                a.metric("Records Analyzed", total)
                b.metric("Hybrid Alerts", alerts)
                c.metric("Below Alert Threshold", benign)
                d.metric(
                    "Alert Rate",
                    f"{alerts / max(total, 1) * 100:.2f}%"
                )

                st.subheader("Hybrid Risk Distribution")

                fig = px.histogram(
                    predictions,
                    x="Hybrid_Risk_Score",
                    nbins=25,
                    title="Distribution of Hybrid Risk Scores"
                )

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Risk Score",
                    yaxis_title="Number of Records"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.subheader("Detection Results")

                styled_predictions = (
                    predictions.style
                    .set_properties(**{
                        "background-color": "#102431",
                        "color": "#e5f4f7",
                        "border-color": "#31505e"
                    })
                    .set_table_styles([
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", "#087e75"),
                                ("color", "#ffffff"),
                                ("font-weight", "bold")
                            ]
                        }
                    ])
                )

                st.dataframe(
                    styled_predictions,
                    use_container_width=True
                )

                csv_data = predictions.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Detection Results",
                    data=csv_data,
                    file_name="x_hids_detection_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Unable to read uploaded CSV: {e}")

# --------------------------------------------------
# KNOWN ATTACK DETECTION
# --------------------------------------------------

elif page == "Known Attack Detection":

    st.title("Known Attack Classification")

    predictions = st.session_state.last_predictions

    if predictions is None:
        st.info("Upload and analyze a CSV first.")
    else:

        st.subheader("Random Forest Classification")

        rf_counts = predictions[
            "RF_Prediction"
        ].value_counts().reset_index()

        rf_counts.columns = ["Attack Class", "Records"]

        st.dataframe(rf_counts, use_container_width=True)

        st.plotly_chart(
            px.bar(
                rf_counts,
                x="Attack Class",
                y="Records",
                title="Random Forest Predictions"
            ).update_layout(template="plotly_dark"),
            use_container_width=True
        )

        st.subheader("XGBoost Classification")

        xgb_counts = predictions[
            "XGB_Prediction"
        ].value_counts().reset_index()

        xgb_counts.columns = ["Attack Class", "Records"]

        st.dataframe(xgb_counts, use_container_width=True)

        st.plotly_chart(
            px.bar(
                xgb_counts,
                x="Attack Class",
                y="Records",
                title="XGBoost Predictions"
            ).update_layout(template="plotly_dark"),
            use_container_width=True
        )

# --------------------------------------------------
# ANOMALY DETECTION
# --------------------------------------------------

elif page == "Anomaly Detection":

    st.title("Unsupervised Anomaly Detection")

    predictions = st.session_state.last_predictions

    if predictions is None:
        st.info("Upload and analyze a CSV first.")
    else:

        c1, c2 = st.columns(2)

        c1.metric(
            "Isolation Forest Anomalies",
            int(predictions["IF_Anomaly"].sum())
        )

        c2.metric(
            "Autoencoder Anomalies",
            int(predictions["AE_Anomaly"].sum())
        )

        anomaly_summary = pd.DataFrame({
            "Detector": [
                "Isolation Forest",
                "Autoencoder"
            ],
            "Anomalies": [
                int(predictions["IF_Anomaly"].sum()),
                int(predictions["AE_Anomaly"].sum())
            ]
        })

        st.plotly_chart(
            px.bar(
                anomaly_summary,
                x="Detector",
                y="Anomalies",
                title="Anomaly Detector Comparison"
            ).update_layout(template="plotly_dark"),
            use_container_width=True
        )

        st.subheader("Anomaly Scores")

        st.dataframe(
            predictions[
                [
                    "IF_Score",
                    "IF_Anomaly",
                    "AE_Reconstruction_Error",
                    "AE_Anomaly"
                ]
            ],
            use_container_width=True
        )

        st.caption(
            "Anomaly outputs indicate deviations from benign-trained "
            "patterns. An anomaly is not automatically a confirmed attack."
        )

# --------------------------------------------------
# HYBRID RISK ANALYSIS
# --------------------------------------------------

elif page == "Hybrid Risk Analysis":

    st.title("Hybrid Threat Risk Engine")

    predictions = st.session_state.last_predictions

    st.write("Original research risk weights:")

    st.json({
        "XGBoost": 0.35,
        "Random Forest": 0.15,
        "Isolation Forest": 0.25,
        "Autoencoder": 0.25
    })

    st.metric("Provisional Alert Threshold", "40 / 100")

    if predictions is None:
        st.info("Upload and analyze a CSV first.")
    else:

        st.subheader("Risk Severity Distribution")

        risk_counts = (
            predictions["Risk_Category"]
            .value_counts()
            .reindex(
                ["Low", "Medium", "High", "Critical"],
                fill_value=0
            )
            .reset_index()
        )

        risk_counts.columns = ["Severity", "Records"]

        st.plotly_chart(
            px.bar(
                risk_counts,
                x="Severity",
                y="Records",
                title="Hybrid Risk Categories"
            ).update_layout(template="plotly_dark"),
            use_container_width=True
        )

        st.subheader("Highest-Risk Records")

        top_risks = predictions.sort_values(
            "Hybrid_Risk_Score",
            ascending=False
        ).head(20)

        st.dataframe(
            top_risks,
            use_container_width=True
        )

        st.caption(
            "The hybrid alert threshold was selected using validation data. "
            "Risk scores are model outputs, not calibrated probabilities."
        )

# --------------------------------------------------
# EXPLAINABLE AI
# --------------------------------------------------

elif page == "Explainable AI":

    st.title("Explainable Artificial Intelligence")

    shap_image = (
        RESULTS / "shap_global_feature_importance.png"
    )

    if shap_image.exists():
        st.subheader("Global SHAP Feature Importance")

        st.image(
            str(shap_image),
            use_container_width=True
        )

    shap_csv = (
        RESULTS / "shap_global_feature_importance.csv"
    )

    if shap_csv.exists():
        st.subheader("Global Feature Importance Data")

        shap_df = pd.read_csv(shap_csv)

        st.dataframe(
            shap_df,
            use_container_width=True
        )

    individual_csv = (
        RESULTS / "shap_individual_explanation.csv"
    )

    if individual_csv.exists():
        st.subheader("Individual XGBoost Explanation")

        individual_df = pd.read_csv(individual_csv)

        st.dataframe(
            individual_df,
            use_container_width=True
        )

    st.caption(
        "SHAP explains the XGBoost classifier, not the complete "
        "hybrid risk score. SHAP values do not establish causality."
    )

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Experimental Model Performance")

    comparison_path = (
        RESULTS / "x_hids_final_model_comparison.csv"
    )

    if comparison_path.exists():

        comparison = pd.read_csv(comparison_path)

        st.dataframe(
            comparison,
            use_container_width=True
        )

        numeric_columns = comparison.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            selected_metric = st.selectbox(
                "Select performance metric",
                numeric_columns
            )

            fig = px.bar(
                comparison,
                x=comparison.columns[0],
                y=selected_metric,
                title=f"Model Comparison — {selected_metric}"
            )

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Model",
                yaxis_title=selected_metric
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:
        st.warning("Model comparison file not found.")

    st.caption(
        "Multiclass classification metrics and binary anomaly/hybrid "
        "metrics use different evaluation definitions. Interpret them "
        "separately."
    )

# --------------------------------------------------
# ALERT HISTORY
# --------------------------------------------------

elif page == "Alert History":

    st.title("Alert History")

    history = st.session_state.alert_history

    if history:

        history_df = pd.DataFrame(history)

        st.metric("Session Alerts", len(history_df))

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.download_button(
            "Download Alert History",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="x_hids_alert_history.csv",
            mime="text/csv"
        )

        if st.button("Clear Session Alert History"):
            st.session_state.alert_history = []
            st.rerun()

    else:
        st.info(
            "No alerts generated in this session. "
            "Run CSV detection to populate alert history."
        )

    st.caption(
        "This version stores alerts in Streamlit session state. "
        "History is not persistent across application restarts."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "X-HIDS | MTech Research Project | CIC-IDS2017 | "
    "Explainable Hybrid AI Intrusion Detection"
)
