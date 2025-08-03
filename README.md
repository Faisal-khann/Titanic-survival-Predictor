## Survival Prediction & Exploratory Analysis -- webApp

This project presents a comprehensive end-to-end machine learning workflow using the iconic Titanic dataset from Kaggle, which is a classic binary classification task aimed at predicting passenger survival. The goal is to build a robust, interpretable, and deployable model capable of estimating the likelihood of survival based on a variety of passenger features. It is complete end-to-end machine learning project that includes data cleaning, feature engineering, model comparison, hyperparameter tuning, and deployment using Streamlit.

<h2><b>Business Problem Statement</b></h2>

Given passenger data such as age, gender, passenger class, fare, and familySize, can we develop a predictive model to estimate the likelihood of survival?<br>
The goal is to uncover patterns that can inform and optimize passenger safety protocols and emergency evacuation strategies in modern transport systems.

<hr>

<h2><b>Real-World Application</b></h2>

Although Titanic is a historical event, this analysis simulates a real-world problem relevant to:

<ul>
  <li><b>Cruise ship safety planning</b></li>
  <li><b>Passenger risk profiling</b></li>
  <li><b>Emergency evacuation strategy</b></li>
  <li><b>Insurance underwriting</b></li>
</ul>

<hr>

<h2><b>Objective</b></h2>

<ul>
  <li>Analyze patterns in the Titanic dataset using <b>exploratory data analysis (EDA)</b></li>
  <li>Handle <b>missing data</b> and prepare features for modeling</li>
  <li>Build a <b>classification model (Random Forest)</b> to predict passenger survival</li>
  <li>Evaluate model performance and interpret <b>feature importance</b></li>
</ul>

<hr>

<h2><b>Business Value</b></h2>

<ul>
  <li>Identify the <b>key factors</b> that influence passenger survival</li>
  <li>Apply <b>data-driven insights</b> to improve safety procedures in transport systems</li>
  <li>Showcase the ability to apply <b>data science skills</b> to real-world business problems</li>
</ul>

</div>


## Features

- Full Exploratory Data Analysis (EDA)
- Feature Engineering (`FamilySize`, `IsAlone`)
- Data preprocessing: Handling missing values, encoding, scaling
- Model Comparison: Logistic Regression, SVM, Random Forest, KNN, Decision Tree
- Hyperparameter tuning using GridSearchCV
- Evaluation: Accuracy, Confusion Matrix, Classification Report
- Visualizations: Age distribution, survival by class/gender, etc.
- Deployment via **Streamlit**

## Resources

- Full Jupyter Notebook: [`Workspace.ipynb`](Workspace.ipynb)
- Live App: [`titanic_app.py`](https://faisal-khann-titanic-survival-predictor-app-hywqlp.streamlit.app/)
- 

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
## Author

**Faisal Khan**  
*Data Analyst | Python | Machine learnig | Data Science*

## Contact

For any questions, collaboration opportunities, or project-related inquiries, feel free to reach out:

- 📧 [Email](mailto:thisside.faisalkhan@example.com)  
- 💼 [LinkedIn](http://www.linkedin.com/in/faisal-khan-332b882bb)

Let’s connect and build something impactful!

---
## 📌 License

This project is for educational purposes.

---
> Made with ❤️ using Jupyter Notebook & Power 



