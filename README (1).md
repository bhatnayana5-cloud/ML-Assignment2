# Heart Disease Prediction using Machine Learning Classifiers

## a. Problem statement
The goal is to predict whether a patient has heart disease (binary classification: 1 = heart disease, 0 = normal) from clinical attributes such as age, sex, chest pain type, resting blood pressure, cholesterol, maximum heart rate and ST-segment measurements. Early and accurate screening helps doctors prioritise high-risk patients for further tests and treatment. Five classification models are trained and compared on the same train/test split, and the best one is served through a Streamlit web app.

## b. Dataset description
- Source: <Kaggle/UCI link to your heart dataset>
- Instances: <n>  | Features: <n>  | Target: <target column name>, 2 classes (0 = no heart disease, 1 = heart disease)
- Preprocessing: <missing values handled (e.g. none / imputed)>, categorical features encoded (one-hot / label encoding), numerical features scaled with StandardScaler (needed for Logistic Regression, kNN and SVM), stratified train/test split. Feature order is saved in `feature_columns.pkl` so the Streamlit app uses the same columns as training.

## c. GitHub repository
<(https://github.com/bhatnayana5-cloud/ML-Assignment2)>

Streamlit app: <ml-assignment2-7zcyc9cab7sbmjjjl36exb.streamlit.app>

Repository contents: `app.py`, `train_models.py`, `heart.csv`, `test_data.csv`, `metrics_table.csv`, `feature_columns.pkl` and the trained models (`logistic_regression.pkl`, `decision_tree.pkl`, `knn.pkl`, `naive_bayes.pkl`, `support_vector_machine.pkl`).

## d. Models used
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8859 | 0.9297 | 0.8716 | 0.9314 | 0.9005 | 0.7694 |
| Decision Tree | 0.7935 | 0.7944 | 0.8200 | 0.8039 | 0.8119 | 0.5832 |
| kNN | 0.8967 | 0.9423 | 0.8952 | 0.9216 | 0.9082 | 0.7907 |
| Naive Bayes | 0.9130 | 0.9451 | 0.9300 | 0.9118 | 0.9208 | 0.8246 |
| Support Vector Machine | 0.9022 | 0.9443 | 0.8818 | 0.9510 | 0.9151 | 0.8033 |

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline (accuracy 0.8859, AUC 0.9297). Its recall of 0.9314 is high, so it misses few patients with heart disease, but precision (0.8716) is slightly lower, meaning a few more false alarms. |
| Decision Tree | The weakest model on every metric (accuracy 0.7935, AUC 0.7944, MCC 0.5832). A single tree overfits the training data and generalises poorly; the low AUC shows it separates the two classes less reliably than the other models. |
| kNN | Performs well (accuracy 0.8967, AUC 0.9423) after feature scaling. Balanced precision (0.8952) and recall (0.9216), but it is slower at prediction time and sensitive to the choice of k. |
| Naive Bayes | The best overall model: highest accuracy (0.9130), AUC (0.9451), precision (0.9300), F1 (0.9208) and MCC (0.8246). Despite its independence assumption it works very well on this dataset, and it is fast and simple. |
| Support Vector Machine | Very close to the best (accuracy 0.9022, AUC 0.9443) and has the highest recall (0.9510), so it catches the most heart disease cases. Its precision (0.8818) is a little lower than Naive Bayes. |

**Overall winner:** Naive Bayes. It scores the best on accuracy, AUC, precision, F1 and MCC, and MCC (0.8246) is the most reliable single summary of binary classification quality. SVM is a close second and is the better choice if maximising recall (not missing any patient) is the priority; Decision Tree performed worst.
