import streamlit as st
import pandas as pd
import utils.data_import as data_import
import utils.prediction as pred
import utils.evaluation_metrix as em
import pages.result as rp

st.set_page_config(
    page_title="Mobile price Prediction App",
    layout="wide"
)

test_file_path = "test.csv"
X_test,y_test = None, None
st.title("Welcome to Mobile pricing prediction model !!!")
st.text("Download the test data to predict the price range of mobile phones")
with open(test_file_path, "rb") as f:
    st.download_button(
        "Download CSV",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )
st.text("Select the model using that you want to predict some result")

# Mapping dictionary
mapping = {
    0: "low",
    1: "medium",
    2: "high",
    3: "very high"
}
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    X_test,y_test = data_import.get_feature_target_data(pd.read_csv(uploaded_file))
    # Map the target variable to categorical labels
    y_test = y_test.map(mapping)

else:
    st.write("Please upload a CSV file to proceed.")
#test_data = data_import.get_mobile_test_data()
# Create a dropdown menu for selecting a hobby
model = st.selectbox("Select a model:", ['Logistic Regression', 'Decision Tree', 'kNN', 'Naive Bayes', 'Random Forest', 'XGBoost'])
# A button that displays text when clicked
if st.button("Calculate"):
    if X_test is not None and y_test is not None:
        y_pred, y_prob = pred.predict(model, X_test)   # Get predictions from the model
        y_pred = pd.Series(y_pred).map(mapping) # Map predictions to categorical labels
        metrics, evaluation_marix_fig = em.evaluation_marix(model,y_test, y_pred, y_prob, title=f"")
        # ---- Layout (2 columns) ----
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"<h4 style='text-align:center;'>Confusion Matrix</h4>", unsafe_allow_html=True)
            cm_fig,cm = em.evaluate_confusionMatrix(model,y_test, y_pred, title=f"") # Get confusion matrix figure and data
            st.pyplot(cm_fig)
        with col2:
            st.markdown(f"<h4 style='text-align:center;'>Performance Metrics</h4>", unsafe_allow_html=True)
            st.pyplot(evaluation_marix_fig)
        st.divider()
        col1, col2 = st.columns(2)
        mcm = em.evaluate_multilabeConfusionMatrix(model, y_test, y_pred)
        with col1:
            st.markdown(f"<h4 style='text-align:center;'>Multilabel Confusion Matrix</h4>", unsafe_allow_html=True)
            st.dataframe(mcm, use_container_width=True)
        with col2:
            st.markdown(f"<h4 style='text-align:center;'>Performance Table</h4>", unsafe_allow_html=True)
            st.dataframe(metrics, use_container_width=True)
        results = X_test.copy()     
        results["Actual Price Range"] = pd.Series(y_test).reset_index(drop=True)
        results["Predicted Price Range"] = pd.Series(y_pred).reset_index(drop=True)
        st.divider()
        st.markdown(f"<h4 style='text-align:center;'> Prediction Table</h4>", unsafe_allow_html=True)
        st.write(results)

    else:
        st.write("Error: Please upload a CSV file to proceed.")