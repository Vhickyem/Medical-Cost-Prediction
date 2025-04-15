# 🏥 Medical Cost Prediction Using Machine Learning
## 📌 Project Overview
This project aims to predict individual medical insurance charges based on their personal and lifestyle attributes using various regression models. The objective is to help ensure fairer pricing models, risk profiling and better policy planning in healthcare.

## 🎯 Problem Statement
Medical insurance charges can vary dramatically based on different factors. Insurance providers need to develop models that accurately predict medical costs for their insured populaton based on these factors.

## 🧠 Solution Approach  
* Data Source: [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

* Target Variable: charges (individual medical costs billed by health insurance)

### Models Applied:

* Linear Regression

* Decision Tree Regressor

* Random Forest Regressor (Final Model)

* Support Vector Regressor

### Feature Engineering:

* Added interaction terms (e.g., bmi*smoker, age_bmi, etc.)

* Feature selection using importance scores from GridSearchCV

* Cross-validation for model performance

* Final model evaluation on test data with confidence interval

## 📊 Final Model Performance (Random Forest Regressor)  
🧪 Test RMSE: 4809.75

📈 95% Confidence Interval: [3720.51, 5694.31]  

This means we can predict a patient’s healthcare insurance cost within a margin of roughly ±980 with 95% confidence.
 
## ✅ Key Takeaways
* There is a strong impact of lifestyle choices (especially smoking) on healthcare cost.

* Linear regression improved after simplifying the feature set.

* Feature importance guided more interpretable and effective modeling.

🌐 Visit [My Portfolio](https://www.datascienceportfol.io/VAdaye) for More Projects

🙌 Acknowledgements  
Dataset by [@mirichoi0218](https://www.kaggle.com/datasets/mirichoi0218) on Kaggle


<p align="center">
  <i>©️Improving Lives Through Data</i>
</p>
