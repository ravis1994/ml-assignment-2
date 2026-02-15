# **Problem statement:**

A mobile phone company wants to estimate the price range of the devices it produces. In a highly competitive market, pricing cannot be decided by guesswork, so the company collects sales data from various mobile phones available in the market.

The objective is to discover the relationship between different features of a mobile phone (such as **RAM, internal memory, battery power**, etc.) and its selling price. Instead of predicting the exact price, the goal is to predict the  **price range** , which indicates how high or low the price is based on the phone’s specifications.

The dataset is used to **predict the price range of a mobile phone** based on its specifications (not the exact price). It is a **multiclass classification** problem.

Price range categories typically mean:

* 0 → Low cost
* 1 → Medium cost
* 2 → High cost
* 3 → Very high cost

# **Dataset description:**

**Dataset Overview:**

* About **2000 training records**
* Around **20 input features**
* **1 target column:** `price_range`

  * 0 = Low cost
  * 1 = Medium cost
  * 2 = High cost
  * 3 = Very high cost
* Clean dataset with mostly numerical and binary features
* RAM, battery power, and resolution strongly affect price range.

**Main Features:**

* **Performance:** RAM, battery_power, clock_speed, n_cores
* **Storage:** int_memory
* **Display:** px_height, px_width, sc_h, sc_w
* **Camera:** pc, fc
* **Connectivity:** bluetooth, wifi, 3G, 4G, dual_sim
* **Physical:** mobile_wt, m_dep, talk_time

# **Models used:**

Comparison Table with the evaluation metrics calculated for all the 6 models as below:

| ML Model Name                 | Accuracy | Precision | Recall | F1 Score | MCC   | AUC   |
| :---------------------------- | -------- | --------- | ------ | -------- | ----- | ----- |
| **Logistic Regression** | 0.743    | 0.745     | 0.743  | 0.744    | 0.657 | 0.579 |
| **Decision Tree**       | 0.847    | 0.848     | 0.847  | 0.846    | 0.796 | 0.579 |
| **kNN**                 | 0.947    | 0.946     | 0.947  | 0.946    | 0.929 | 0.514 |
| **Naive Bayes**         | 0.823    | 0.833     | 0.823  | 0.825    | 0.767 | 0.584 |
| **Random Forest**       | 0.880    | 0.880     | 0.880  | 0.880    | 0.840 | 0.579 |
| **XGBoost**             | 0.900    | 0.901     | 0.900  | 0.900    | 0.867 | 0.602 |

Observations on the performance of each model on the chosen dataset:

| ML Model Name                 | Observation about model performance                                                                                                                                                                                                                                                   |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Lowest accuracy (≈74%) and MCC among all models, indicating limited ability to capture complex patterns. Confusion matrix shows higher misclassification between low, high, and very high classes, suggesting linear decision boundaries are insufficient for this dataset.          |
| **Decision Tree**       | Moderate performance (≈84.7% accuracy) with improved MCC compared to Logistic Regression and Naive Bayes. Captures nonlinear relationships but shows some misclassification in low and medium classes, which may indicate mild overfitting or limited generalization.                |
| **kNN**                 | Best overall performance with highest accuracy (≈94.7%), F1-score, and MCC, indicating strong class separability in the dataset. Very low misclassification across all classes, though relatively lower AUC and higher prediction cost may limit scalability on very large datasets. |
| **Naive Bayes**         | Good baseline performance (≈82.3% accuracy) and fast training, performing particularly well on the medium class. However, independence assumptions between features likely reduce accuracy compared to tree-based and ensemble models.                                               |
| **Random Forest**       | Strong and stable performance (≈88% accuracy) with balanced precision, recall, and MCC. Ensemble learning reduces variance and improves generalization, with fewer misclassifications across all classes compared to single Decision Tree.                                           |
| **XGBoost**             | High performance (≈90% accuracy) and strong MCC, with the best AUC among the models. Handles class boundaries effectively and shows low misclassification, indicating good generalization and capability to model complex relationships.                                             |

### Final Conclusion

Among all the evaluated models, kNN achieved the highest accuracy, F1-score, and Matthews Correlation Coefficient, indicating that the dataset has well-separated feature patterns that benefit distance-based learning. XGBoost and Random Forest also demonstrated strong and reliable performance, showing good generalization and robustness due to ensemble learning. Decision Tree and Naive Bayes provided moderate performance and can serve as interpretable or fast baseline models. Logistic Regression showed the lowest performance, suggesting that linear decision boundaries are insufficient for this problem. Overall, kNN is the most suitable model for this dataset, while ensemble models remain strong alternatives for scalable and production-oriented scenarios.
