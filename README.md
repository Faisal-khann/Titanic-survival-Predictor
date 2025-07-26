## Survival Prediction & Exploratory Analysis -- webApp

This project presents a comprehensive end-to-end machine learning workflow using the iconic Titanic dataset from Kaggle, which is a classic binary classification task aimed at predicting passenger survival. The goal is to build a robust, interpretable, and deployable model capable of estimating the likelihood of survival based on a variety of passenger features. It is complete end-to-end machine learning project that includes data cleaning, feature engineering, model comparison, hyperparameter tuning, and deployment using Streamlit.


## Features

- Full Exploratory Data Analysis (EDA)
- Feature Engineering (`FamilySize`, `IsAlone`)
- Data preprocessing: Handling missing values, encoding, scaling
- Model Comparison: Logistic Regression, SVM, Random Forest, KNN, Decision Tree
- Hyperparameter tuning using GridSearchCV
- Evaluation: Accuracy, Confusion Matrix, Classification Report
- Visualizations: Age distribution, survival by class/gender, etc.
- Deployment via **Streamlit**

## Live App

👉 [Click here to try the app](https://faisal-khann-titanic-survival-predictor-app-hywqlp.streamlit.app/)

## Project Structure
    ├── titanic_app.py # Streamlit app
    ├── titanic_model.pkl # Trained ML model
    ├── requirements.txt # Python dependencies
    ├── titanic_training.ipynb # EDA + training notebook
    └── README.md # This file

## Tech Stack

- Python
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- Streamlit

## Project Workflow

The project follows a complete data science pipeline that includes:

- **Data Cleaning and Preprocessing**  
  Handling missing values, dropping irrelevant features, and preparing the dataset for modeling.

- **Feature Engineering for Model Optimization**  
  Transforming categorical variables, creating new features like FamilySize, and encoding data for model compatibility.

- **Model Comparison using Cross-Validation and Accuracy Metrics**  
  Evaluating models including Logistic Regression, Random Forest, and SVM using `cross_val_score` and classification metrics.

- **Hyperparameter Tuning for Performance Enhancement**  
  Optimizing model parameters using techniques like Grid Search for improved accuracy.

- **Deployment of the Final Model using Streamlit**  
  Building a streamlit web app to make predictions interactively based on user input and perform end-to-end exploratory data analysis.

<!--## Sample Screenshot

(Add a screenshot of your Streamlit app here if you want)

--->

## Contact

Faisal Khan
[LinkedIn](http://www.linkedin.com/in/faisal-khan-332b882bb) | [GitHub](https://github.com/Faisal-khann)

## 📌 License

This project is for educational purposes.


