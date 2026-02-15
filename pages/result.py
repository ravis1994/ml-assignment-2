import streamlit as st
import pandas as pd
import numpy as np
import json
from utils.evaluation_metrix import plot_confusion_matrix, plot_evaluation_metrics

file_path = "model_results.json"
st.set_page_config(
    page_title="Mobile price Result Metrics Dashboard",
    layout="wide"
)
st.title("Mobile Pricing  - Result Metrics Dashboard")

model = ['Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost']

try:
    with open(file_path, "r") as f:
        model_results_data = json.load(f)
except FileNotFoundError:
            st.write(f"matrix data not found.")
table_data = {}
for model_name, model_data in model_results_data.items():
    table_data[model_name] = model_data["evaluation_metrics"]

df_metrics = pd.DataFrame.from_dict(table_data, orient="index").round(3)
st.subheader("Evaluation Metrics Table")
st.dataframe(df_metrics, use_container_width=True)

for m in model:
    st.divider()
    st.markdown(f"<h3 style='text-align:center;'>{m}</h3>", unsafe_allow_html=True)
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        try: 
            fig = plot_confusion_matrix(m, np.array(model_results_data[m]["confusion_matrix"]), title=f"Confusion Matrix")
            st.pyplot(fig)
            #st.dataframe(model_results_data[m]["multilabel_confusion_matrix"], use_container_width=True)
        except KeyError:
            st.write(f"Confusion matrix data for {m} not found.")
    with col2:
        try: 
            metrics = model_results_data[m]["evaluation_metrics"]
            fig = plot_evaluation_metrics(m, metrics, title=f"Evaluation Metrics")
            st.pyplot(fig)
        except KeyError:
            st.write(f"Evaluation metrics data for {m} not found.")
    col1, col2 = st.columns(2)
    with col1:
        try: 
            st.markdown(f"<p style='text-align:center;'>Multilabel Confusion Matrix</p>", unsafe_allow_html=True)
            st.dataframe(model_results_data[m]["multilabel_confusion_matrix"], use_container_width=True)
        except KeyError:
            st.write(f"Multilabel confusion matrix data for {m} not found.")
    with col2:
        try: 
            st.markdown(f"<p style='text-align:center;'>Evaluation Metrics</p>", unsafe_allow_html=True)
            st.dataframe(model_results_data[m]["evaluation_metrics"], use_container_width=True)
        except KeyError:
            st.write(f"Evaluation metrics data for {m} not found.")
       

