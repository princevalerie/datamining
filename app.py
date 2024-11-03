# app.py
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import KFold, cross_val_score

# Load dataset
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
DecisionTree = DecisionTreeClassifier(criterion='entropy', max_depth=2)
kf = KFold(n_splits=10, shuffle=True, random_state=1)  # Using 10-fold cross-validation
cv_scores_dt = cross_val_score(DecisionTree, x_data_normalized, y_target, cv=kf, scoring='accuracy')

# Train Decision Tree for visualization
DecisionTree.fit(x_data_normalized, y_target)

# Streamlit App
st.title("Heart Attack Prediction App - Decision Tree")

# Display cross-validation accuracy
st.subheader("Decision Tree Cross-Validation Accuracy")
st.write(f"Average CV Accuracy: {np.mean(cv_scores_dt):.2f}")
st.write(f"Standard Deviation: {np.std(cv_scores_dt):.2f}")

# Plot and display the Decision Tree
st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(DecisionTree, filled=True, feature_names=columns_to_normalize, class_names=["No Heart Disease", "Heart Disease"], ax=ax)
st.pyplot(fig)

# Input features for prediction
st.header("Enter Input Values:")

age = st.slider("Age", 21, 91, 55, step=1)
gender_options = {"Female [0]": 0, "Male [1]": 1}
gender = st.radio("Gender", list(gender_options.keys()))
impluse = st.slider("Impluse", 36, 114, 75, step=1)
pressurehight = st.slider("Pressure High", 65, 193, 126, step=1)
pressurelow = st.slider("Pressure Low", 38, 105, 72, step=1)
glucose = st.slider("Glucose", 35, 279, 130, step=1)
kcm = st.slider("KCM", 0.321, 11.94, 3.11, step=0.01)
troponin = st.slider("Troponin", 0.002, 0.192, 0.022, step=0.001)

if st.button("Predict"):
    # Prepare input for prediction
    input_values = np.array([age, gender_options[gender], impluse, pressurehight, pressurelow, glucose, kcm, troponin]).reshape(1, -1)
    input_values_normalized = scaler.transform(input_values)
    input_df = pd.DataFrame(input_values_normalized, columns=columns_to_normalize)
    
    # Predict with Decision Tree
    prediction_dt = DecisionTree.predict(input_df)

    # Display prediction
    st.subheader("Prediction Result:")
    if prediction_dt[0] == 1:
        st.write("Patient is predicted to have heart disease. == [ Positive ]")
    else:
        st.write("Patient isn't predicted to have heart disease. == [ Negative ]")
