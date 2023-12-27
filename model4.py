# app.py
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score

# Import dataset
data = pd.read_csv("heart_attack.csv")
data.drop_duplicates(inplace=True)

# Convert categorical data to numeric
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

# Drop outliers
columns_to_drop_outliers = ['age', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']
for column_name in columns_to_drop_outliers:
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_iqr = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    data = data.drop(outlier_iqr.index)

# Normalize data
columns_to_normalize = ['age', 'gender', 'impluse', 'pressurehight', 'pressurelow', 'glucose', 'kcm', 'troponin']
x_data = data[columns_to_normalize]
y_target = data['class']
scaler = MinMaxScaler()
x_data_normalized = scaler.fit_transform(x_data)
x_data_normalized = pd.DataFrame(x_data_normalized, columns=columns_to_normalize)

# Decision Tree model
X_dt = x_data_normalized.copy()
y_dt = y_target.copy()
DecisionTree = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                                      max_depth=4, max_features=None, max_leaf_nodes=None,
                                      min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2,
                                      min_weight_fraction_leaf=0.0, splitter='best')
DecisionTree.fit(X_dt, y_dt)

# KNN model
best_n_neighbors = 5  # Assume 5 as the best value based on previous code
best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
best_knn.fit(X_dt, y_dt)

# Naive Bayes model
classifier = GaussianNB()
classifier.fit(X_dt, y_dt)

# Streamlit App
st.title("Heart Attack Prediction App")

# Sidebar
st.sidebar.header("Options")

# Main content
st.header("Enter Input Values:")



age = st.slider("Age", 21, 91, 55, step=1)
# gender_options = [0, 1]
# gender = st.radio("Gender", gender_options)
# Explanation for gender options
gender_options = {"Female [0]": 0, "Male [1]": 1}
gender = st.radio("Gender", list(gender_options.keys()))


impluse = st.slider("Impluse", 36, 114, 75, step=1)
pressurehight = st.slider("Pressure High", 65, 193, 126, step=1)
pressurelow = st.slider("Pressure Low", 38, 105, 72, step=1)
glucose = st.slider("Glucose", 35, 279, 130, step=1)
kcm = st.slider("KCM", 0.321, 11.94, 3.11, step=0.01)
troponin = st.slider("Troponin", 0.002, 0.192, 0.022, step=0.001)



# ...
if st.button("Predict"):
    st.sidebar.subheader("Prediction Results:")

    input_values = np.array([age, gender_options[gender], impluse, pressurehight, pressurelow, glucose, kcm, troponin]).reshape(1, -1)
    input_values_normalized = scaler.transform(input_values)
    input_df = pd.DataFrame(input_values_normalized, columns=columns_to_normalize)

    prediction_dt = DecisionTree.predict(input_df)
    prediction_knn = best_knn.predict(input_df)
    prediction_nb = classifier.predict(input_df)

    st.sidebar.write("Decision Tree Prediction:")
    if prediction_dt[0] == 1:
        st.sidebar.write(f"Patients are predicted to have heart disease. == [ Positive ]")
    else:
        st.sidebar.write(f"Patients aren't predicted to have heart disease. == [ Negative ]")

    st.sidebar.write("KNN Prediction:")
    if prediction_knn[0] == 1:
        st.sidebar.write(f"Patients are predicted to have heart disease. == [ Positive ]")
    else:
        st.sidebar.write(f"Patients aren't predicted to have heart disease. == [ Negative ]")

    st.sidebar.write("Naive Bayes Prediction:")
    if prediction_nb[0] == 1:
        st.sidebar.write(f"Patients are predicted to have heart disease. == [ Positive ]")
    else:
        st.sidebar.write(f"Patients aren't predicted to have heart disease. == [ Negative ]")

    # Evaluate models
    st.sidebar.subheader("Model Evaluation:")
    
    # Decision Tree
    y_dt_pred = DecisionTree.predict(X_dt)
    accuracy_dt = accuracy_score(y_dt, y_dt_pred)

    # KNN
    y_knn_pred = best_knn.predict(X_dt)
    accuracy_knn = accuracy_score(y_dt, y_knn_pred)

    # Naive Bayes
    y_nb_pred = classifier.predict(X_dt)
    accuracy_nb = accuracy_score(y_dt, y_nb_pred)

    # Display accuracy scores
    st.sidebar.subheader("Model Accuracy:")
    st.sidebar.write(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
    st.sidebar.write(f"KNN Accuracy: {accuracy_knn:.2f}")
    st.sidebar.write(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")

    if((prediction_dt[0] == 1)&(prediction_knn[0] == 1)&(prediction_nb[0] == 1)):
        accuracy1=(accuracy_dt+accuracy_nb)/2
        st.write(f"Patients are predicted to have heart disease == [ Positive ] , accuraccy {accuracy1:.2f}")
    elif((prediction_dt[0] == 1)&(prediction_nb[0] == 1)):
        accuracy2=(accuracy_dt+accuracy_nb)/2
        st.write(f"Patients are predicted to have heart disease == [ Positive ] , accuraccy {accuracy2:.2f}")
    elif((prediction_dt[0] == 1)&(prediction_knn[0] == 1)):
        accuracy3=(accuracy_dt+accuracy_knn)/2
        st.write(f"Patients are predicted to have heart disease == [ Positive ] , accuraccy {accuracy3:.2f}")
    elif((prediction_nb[0] == 1)&(prediction_knn[0] == 1)):
        accuracy4=(accuracy_dt+accuracy_knn)/2
        st.write(f"Patients are predicted to have heart disease == [ Positive ] , accuraccy {accuracy4:.2f}")

    if((prediction_dt[0] == 0)&(prediction_knn[0] == 0)&(prediction_nb[0] == 0)):
        accuracy1_=(accuracy_dt+accuracy_nb)/2
        st.write(f"Patients aren't predicted to have heart disease == [ Negative ] , accuraccy {accuracy1_:.2f}")
    elif((prediction_dt[0] == 0)&(prediction_nb[0] == 0)):
        accuracy2_=(accuracy_dt+accuracy_nb)/2
        st.write(f"Patients aren't predicted to have heart disease == [ Negative ] ,accuraccy {accuracy2_:.2f}")
    elif((prediction_dt[0] == 0)&(prediction_knn[0] == 0)):
        accuracy3_=(accuracy_dt+accuracy_knn)/2
        st.write(f"Patients aren't predicted to have heart disease == [ Negative ] ,accuraccy {accuracy3_:.2f}")
    elif((prediction_nb[0] == 0)&(prediction_knn[0] == 0)):
        accuracy4_=(accuracy_dt+accuracy_knn)/2
        st.write(f"Patients aren't predicted to have heart disease == [ Negative ] ,accuraccy {accuracy4_:.2f}")

    

