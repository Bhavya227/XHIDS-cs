# X-HIDS: Explainable Hybrid AI-Based Intrusion Detection System

## 1. Project Overview

X-HIDS is an Explainable Hybrid Artificial Intelligence-Based
Intrusion Detection System designed to identify known network
attacks and investigate potentially unknown or anomalous traffic.

The system combines supervised machine learning, unsupervised
anomaly detection, deep learning, and Explainable AI.

It provides a Streamlit dashboard for network traffic analysis,
attack detection, risk assessment, and model explanations.

## 2. Objectives

- Detect known network attacks using supervised learning.
- Identify anomalous traffic using unsupervised learning.
- Explore potential zero-day or previously unseen attack patterns.
- Combine multiple models into a hybrid detection framework.
- Reduce false alarms through hybrid risk analysis.
- Provide interpretable predictions using SHAP.
- Present detection results through an interactive dashboard.

## 3. System Architecture

The X-HIDS workflow consists of:

1. Network traffic dataset acquisition.
2. Data cleaning and preprocessing.
3. Feature configuration and transformation.
4. Supervised model training.
5. Unsupervised anomaly detection.
6. Autoencoder-based anomaly analysis.
7. Hybrid risk scoring.
8. SHAP-based explanation.
9. Streamlit dashboard visualization.

## 4. Dataset

**Dataset:** CIC-IDS2017

The dataset was developed by the Canadian Institute for
Cybersecurity at the University of New Brunswick.

It contains benign traffic and multiple categories of
cyberattacks, including denial-of-service, distributed
denial-of-service, brute-force, botnet, port scanning,
and web-based attacks.

The project uses 71 numerical features and 15 traffic classes.

The original dataset is not included in this repository.
Users must obtain it separately and follow its applicable
usage and licensing conditions.

## 5. Machine Learning Models

X-HIDS integrates four main detection models:

| Model | Purpose |
|---|---|
| Random Forest | Supervised multi-class attack classification |
| XGBoost | Supervised attack classification |
| Isolation Forest | Unsupervised anomaly detection |
| Autoencoder | Reconstruction-based anomaly detection |

### Hybrid Detection

The hybrid framework combines supervised predictions and
anomaly scores to calculate a risk score between 0 and 100.

The risk score is used to categorize network traffic into
different severity levels.

The score is a model-derived risk indicator, not a calibrated
probability of an attack.

## 6. Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand
the contribution of input features to XGBoost predictions.

The project includes global feature importance and
individual prediction explanations.

These explanations describe model behavior and should not
be interpreted as causal relationships.

## 7. Experimental Evaluation

The project evaluates the models using classification
metrics, confusion matrices, per-class detection results,
and anomaly detection metrics.

The experiments include an unseen-category evaluation in
which DoS Hulk traffic is held out from supervised training.

This experiment investigates generalization to an unseen
attack category. It does not establish guaranteed detection
of arbitrary real-world zero-day attacks.

Refer to the `results/` directory for the detailed
experimental outputs.

## 8. Streamlit Dashboard

The interactive dashboard provides:

- Command Center
- CSV Detection
- Known Attack Detection
- Anomaly Detection
- Hybrid Risk Analysis
- Explainable AI
- Model Performance
- Alert History

### Run Locally

Clone the repository:

```bash
git clone https://github.com/Bhavya227/XHIDS-cs.git
cd XHIDS-cs
```

Install dependencies:

```bash
pip install -r dashboard/requirements.txt
```

Launch the dashboard from the repository root:

```bash
streamlit run dashboard/app.py
```

Open the local URL displayed in the terminal.

## 9. Repository Structure

```text
XHIDS-cs/
│
├── dashboard/
│   ├── app.py
│   ├── predictor.py
│   ├── requirements.txt
│   ├── models/
│   ├── preprocessing/
│   ├── results/
│   └── test_data/
│
├── models/
├── preprocessing/
├── results/
│
├── X-HIDS_Main_Research_Notebook.ipynb
└── README.md
```

## 10. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- PyTorch
- SHAP
- Streamlit
- Plotly
- Joblib
- Google Colab
- Git and GitHub

## 11. Limitations

- Detection performance depends on dataset quality and
  the representativeness of training data.
- An unseen attack category is not equivalent to every
  possible zero-day attack.
- Hybrid risk scores are not calibrated attack probabilities.
- SHAP explanations currently focus on the XGBoost model.
- Performance on real-world live network traffic requires
  further validation.

## 12. Future Enhancements

- Live network traffic integration.
- Continuous model monitoring.
- Improved zero-day detection and validation.
- Real-time alert notification.
- Enhanced explainability across all hybrid components.
- Deployment on a cloud platform.

## 13. Research Reference

Sharafaldin, I., Lashkari, A. H., and Ghorbani, A. A. (2018).

"Toward Generating a New Intrusion Detection Dataset and
Intrusion Traffic Characterization."

Proceedings of the International Conference on Information
Systems Security and Privacy (ICISSP).

Dataset: CIC-IDS2017, Canadian Institute for Cybersecurity,
University of New Brunswick.

---

**Project:** X-HIDS  
**Domain:** Cybersecurity and Network Intrusion Detection  
**Repository:** https://github.com/Bhavya227/XHIDS-cs
