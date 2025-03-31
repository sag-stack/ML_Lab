# ML-Lab-Showcase

## Overview
This document presents a summary of three machine learning projects focusing on different techniques: Decision Trees, Ensemble Regression, and K-Nearest Neighbors (KNN) Classification. Each project follows structured steps including data preprocessing, model training, evaluation, and key observations.

---

## Decision Tree Classification

### Objective
To predict wine quality based on various chemical properties using a Decision Tree model.

### Steps Followed
1. **Data Preprocessing**
   - Loaded dataset (`Decision_Tree_train.csv`).
   - Handled missing values by filling them with column means.
   
2. **Feature Selection**
   - Used `SelectKBest` with `f_classif` to select the top 5 most significant features:
     - Volatile Acidity
     - Chlorides
     - Total Sulfur Dioxide
     - Density
     - Alcohol

3. **Model Training**
   - Data split into training (80%) and testing (20%) sets.
   - Trained a Decision Tree classifier on selected features.

4. **Evaluation Metrics**
   - **Confusion Matrix**: Displayed class-wise performance.
   - **F1 Score**: 0.5385, indicating moderate classification performance.
   - **Classification Report**: The model struggled with underrepresented classes (3 & 9) but performed well on majority classes (5 & 6).

5. **Test Data Prediction**
   - Preprocessed `Decision_Tree_test.csv` similarly.
   - Predictions saved in `submission.csv`.

### Key Findings
- Model struggled with overlapping classes (e.g., 5 & 6).
- Could improve with hyperparameter tuning (max depth, minimum samples per leaf) or advanced models like Random Forest.

---

## Ensemble Regression

### Objective
To predict a continuous target variable (`y`) using ensemble regression techniques.

### Steps Followed
1. **Data Preprocessing**
   - Loaded dataset (`Ensemble_Reg_train.csv`).
   - Standardized features using `StandardScaler`.

2. **Train-Test Split**
   - Dataset split into 80% training and 20% testing.

3. **Model Selection & Tuning**
   - Used **GridSearchCV** to optimize:
     - **RandomForestRegressor** (`max_depth=5, n_estimators=300`, RMSE=52.15)
     - **SVR** (`C=1, epsilon=0.1, kernel=linear`, RMSE=51.75)

4. **Ensemble Models Tested**
   - **Voting Regressor** (RandomForest + SVR): Best performer
     - Test RMSE: 60.48
     - Cross-validation RMSE: 50.47
   - **Bagging Regressor** (RandomForest base model)
     - Test RMSE: 62.36
     - Cross-validation RMSE: 52.12
   - **Stacking Regressor** (RandomForest + SVR with Linear Regression final estimator)
     - Test RMSE: 61.45
     - Cross-validation RMSE: 50.80

5. **Final Model & Predictions**
   - **Voting Regressor** chosen for final predictions on `Ensemble_Reg_test.csv`.
   - Results saved in `submission.csv`.

### Key Findings
- Ensemble methods improved performance over individual models.
- Hyperparameter tuning played a crucial role in reducing RMSE.
- Voting Regressor demonstrated the best generalization.

---

## K-Nearest Neighbors (KNN) Classification

### Objective
To classify objects based on mass, width, and height using a KNN classifier.

### Steps Followed
1. **Data Exploration & Preprocessing**
   - Loaded dataset (`KNN_train.csv`).
   - Features: `mass, width, height` (ID & label removed).
   - Standardized using `StandardScaler`.

2. **Model Training**
   - Used **KNN classifier** with `n_neighbors=1`.

3. **Model Evaluation**
   - Achieved **100% accuracy** on training data.
   - Classification report showed perfect precision, recall, and F1-score.

4. **Testing & Predictions**
   - Predictions made on `KNN_test.csv`.
   - Results saved in `submission.csv`.

### Key Findings
- The model perfectly classified training data but may overfit.
- Testing performance needs evaluation for generalization.

---

## Logistic Regression 

### **Objective:**
To classify data points into distinct categories using a logistic regression model.

### **Preprocessing & Feature Selection:**
- Performed data cleaning, handling missing values.
- Applied feature scaling to standardize the dataset.
- Selected the most relevant features using correlation analysis.

### **Model Implementation:**
- Utilized scikit-learn’s `LogisticRegression`.
- Split data into training and testing sets.
- Trained the model on the training data.

### **Evaluation Metrics:**
- **Accuracy:** Measured correct predictions.
- **Confusion Matrix:** Visualized classification performance.
- **Precision, Recall, F1-Score:** Analyzed the balance between false positives and false negatives.

### **Results:**
- The model performed well on the dataset but showed some limitations with class imbalance.

---

## Multi-Layer Perceptron (MLP) 

### **Objective:**
To implement a neural network for classification using the MNIST dataset.

### **Preprocessing & Feature Engineering:**
- Normalized pixel values of images between 0 and 1.
- Flattened image data into feature vectors.

### **Model Architecture:**
- Used `MLPClassifier` from scikit-learn.
- Hidden layers: Configured with different neuron counts.
- Activation function: ReLU for hidden layers, softmax for output.
- Optimizer: Adam, with learning rate tuning.

### **Evaluation Metrics:**
- **Accuracy Score:** Compared with logistic regression results.
- **Loss Curve:** Tracked training process.
- **Confusion Matrix:** Identified misclassified digits.

### **Results:**
- Achieved higher accuracy than logistic regression.
- Required hyperparameter tuning for optimal performance.

---

## Multi-Linear Regression Model

### **Objective:**
To predict a continuous target variable based on multiple independent variables.

### **Preprocessing & Feature Selection:**
- Checked for multicollinearity using VIF (Variance Inflation Factor).
- Standardized numerical features for consistency.

### **Model Implementation:**
- Applied `LinearRegression` from scikit-learn.
- Trained the model using a training dataset.
- Predicted values for test data.

### **Evaluation Metrics:**
- **Mean Squared Error (MSE):** Assessed prediction errors.
- **R-squared Score:** Measured variance explanation.
- **Residual Analysis:** Ensured assumptions of linear regression were met.

### **Results:**
- Provided reasonable predictions but required feature engineering for improvement.
- Outliers and non-linearity affected accuracy.

---

## Conclusion
This showcase presents three machine learning applications with distinct approaches:
1. **Decision Tree for classification**: Moderate performance with potential improvements.
2. **Ensemble Regression**: Voting Regressor emerged as the best model for prediction.
3. **KNN Classification**: Highly accurate but requires validation on test data.
4. **Logistic Regression**: Performed well but had class imbalance issues.
5. **MLP**: Outperformed logistic regression but needed careful tuning.
6. **Multi-Linear Regression**: Worked for continuous data but was sensitive to feature selection.


This ML-Lab-Showcase demonstrates key ML techniques and their effectiveness in different scenarios, providing a foundation for further research and refinement.

