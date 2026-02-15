from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    multilabel_confusion_matrix,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os
import json

labels = ["low", "medium", "high", "very high"]
def evaluate_confusionMatrix(model, y_test, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    save_model_results(model, "confusion_matrix", cm.tolist(), file_path="model_results.json")
    return plot_confusion_matrix(model, cm, title),cm

def plot_confusion_matrix(model_name, cm, title):
    fig, ax = plt.subplots()
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax)
    ax.set_title(f"{title}")
    return fig

def evaluation_marix(model_name,y_test, y_pred,y_prob, title="Evaluation Metrics"):
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob, multi_class='ovr')
        }
    save_model_results(model_name, "evaluation_metrics", metrics, file_path="model_results.json")
    return metrics, plot_evaluation_metrics(model_name, metrics, title=title)

def plot_evaluation_metrics(model_name, metrics, title="Evaluation Metrics"):
    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values())
    ax.set_title(f"{title}")
    ax.set_ylabel("Score")
    ax.set_xticklabels(metrics.keys(), rotation=45)
    return fig

def evaluate_multilabeConfusionMatrix(model, y_test, y_pred):
    mcm = multilabel_confusion_matrix(y_test,y_pred)
    rows = []
    for i, matrix in enumerate(mcm):
        TN, FP, FN, TP = matrix.ravel()
        rows.append({
        "Class": labels[i],
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN
    })
    df_stats = pd.DataFrame(rows)
    save_model_results(model, "multilabel_confusion_matrix", df_stats.to_dict('records'), file_path="model_results.json")   
    return df_stats

def save_model_results(model_name, metricsName, metrics, file_path="model_results.json"):
    # Step 1: Load existing JSON if file exists
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}
    # Step 2: Update or add model results
    if model_name not in data:
        data[model_name] = {}
    data[model_name][metricsName] = metrics
    # Step 3: Save back to file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"{model_name} {metricsName} results saved successfully.")