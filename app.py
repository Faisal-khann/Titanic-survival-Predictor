import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from PIL import Image

# Load model and dataset
model = joblib.load("final_model.pkl")
df = pd.read_csv("titanic.csv") 

# Streamlit config
st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")
st.title("Titanic Survival Prediction App")

# === Sidebar - Navigation ===
st.sidebar.title("Titanic Survival Predictor ")
img = Image.open('ship.png')
st.sidebar.image(img)
section = st.sidebar.radio("Choose Section", ["Prediction", "EDA"])

# === Prediction Section ===
if section == "Prediction":
    st.header("Titanic Survival Prediction")

    sex = st.selectbox("Sex", ['male', 'female'])
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    age = st.slider("Age", 0, 80, 30)
    fare = st.slider("Fare ($)", 0.0, 250.0, 50.0)
    family_size = st.slider("Family Size (SibSp + Parch + 1)", 1, 11, 1)

    sex = 1 if sex == 'male' else 0

    input_data = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'Fare': fare,
        'FamilySize': family_size,
    }])

    if st.button("Predict Survival"):
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]
        result = "Survived!" if prediction == 1 else "Did Not Survive"
        st.subheader(f"Prediction Result: {result}")
        st.write(f"Survival Probability: **{proba:.2%}**")

# === EDA Section ===
elif section == "EDA":
    st.header("Exploratory Data Analysis")

    # Display business objective
    with st.expander("Business Objective & Real-World Application"):
        st.markdown("""
        ### **Business Objective:**
        To perform an in-depth analysis of the Titanic passenger dataset and build a machine learning model that can accurately predict passenger survival.  
        The goal is to understand key factors influencing survival — such as age, gender, class, and fare — and translate these insights into actionable intelligence.

        ### **Real-World Applications:**
        - **Maritime Safety Planning:** Improve protocols for passenger safety and lifeboat allocation during maritime disasters.
        - **Emergency Response Strategy:** Guide evacuation priorities and crew training using demographic survival patterns.
        - **Insurance & Risk Assessment:** Assist insurers in evaluating travel-related risk based on passenger profiles.
        - **Policy Development:** Inform regulatory decisions regarding safety standards on passenger ships.
        """)    


    eda_type = st.sidebar.radio("EDA Type", ["Univariate Analysis", "Bivariate Analysis"])

    if st.checkbox("Show Raw Data"):
        st.write(df.head())

    # --- Univariate Analysis ---
    if eda_type == "Univariate Analysis":
        st.subheader("Univariate Analysis")

        st.markdown("**Missing Values Heatmap**")
        fig1, ax1 = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax1)
        st.pyplot(fig1)

        st.markdown("**Survival Count**")
        fig2, ax2 = plt.subplots()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        ax2 = sns.countplot(x='Survived', data=df, hue='Survived')
        plt.xticks([0, 1], ['Not Survived', 'Survived'])
        for bars in ax2.containers:
            ax2.bar_label(bars)
        plt.title("Survival Count")
        st.pyplot(fig2)

        survival_rate = df['Survived'].value_counts(normalize=True) * 100
        st.markdown("**Survival Rate (%):**")
        st.write(survival_rate.round(2))

        st.markdown("**Survival Distribution Pie Chart**")
        fig_pie, ax_pie = plt.subplots()
        survival_count = df['Survived'].value_counts()
        ax_pie = survival_count.plot(kind='pie', autopct='%.2f', startangle=90, labels=['Not Survived', 'Survived'], colors=['#ff9999','#66b3ff'])
        ax_pie.set_ylabel('')  # Remove y-axis label
        ax_pie.set_title("Survival Distribution (Pie Chart)")
        st.pyplot(fig_pie)


        st.markdown("**Passenger Gender Distribution**")
        fig_gender, ax_gender = plt.subplots()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        ax_gender = sns.countplot(data=df, x='Sex', hue='Sex')
        for bars in ax_gender.containers:
            ax_gender.bar_label(bars)
        ax_gender.set_title("Gender Distribution")
        st.pyplot(fig_gender)

        gender_rate = df['Sex'].value_counts(normalize=True) * 100
        st.markdown("**Gender Percentage Distribution:**")
        st.write(gender_rate.round(2))


        st.markdown("**Passenger Class Distribution**")
        fig6, ax6 = plt.subplots()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        ax6 = sns.countplot(data=df, x='Pclass', hue='Pclass')
        for bars in ax6.containers:
            ax6.bar_label(bars)
        plt.title("Passenger Class Distribution")
        st.pyplot(fig6)

        Pclass_rate = df['Pclass'].value_counts(normalize=True) * 100
        st.markdown("**Pclass Percentage Distribution:**")
        st.write(Pclass_rate.round(2))

        st.markdown("**Age Distribution**")
        fig_age, ax_age = plt.subplots(figsize=(10, 5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        sns.histplot(df['Age'], bins=30, kde=True, ax=ax_age)
        ax_age.set_title("Age Distribution")
        st.pyplot(fig_age)

        st.markdown("### Conclusion on Age Distribution:")
        st.markdown("""
        - Most passengers were between **20 and 40 years old**.
        - The distribution is slightly **right-skewed**, indicating fewer older passengers.
        - This suggests the dataset has more **young and middle-aged adults**.
        - **Age** could be a useful feature for predicting survival.
        """)


        st.markdown("**Fare Distribution**")
        fig3, ax3 = plt.subplots()
        sns.histplot(df['Fare'], bins=30, kde=True, ax=ax3)
        ax3.set_title("Fare Distribution")
        st.pyplot(fig3)

        st.markdown("### Final Conclusion from Univariate Analysis:")
        st.info(f"""
        - **Most passengers were aged between 20 and 40**, indicating a young-to-middle-aged majority.
        - **Fare distribution** is highly **right-skewed**, with most passengers paying lower fares.
        - **Survival rate** shows that approximately **{survival_rate[1]:.1f}%** of passengers survived the disaster.
        - **Gender distribution** reveals more **males** than females on board (**{gender_rate['male']:.1f}% male**).
        - **Passenger Class** was dominated by **3rd class travelers**, who made up the largest group (**{Pclass_rate[3]:.1f}%**).
        - These univariate insights suggest that **age, fare, class, and gender** are potentially strong features for survival prediction.
        """)



    # --- Bivariate Analysis ---
    elif eda_type == "Bivariate Analysis":
        st.subheader("Bivariate Analysis")

        st.markdown("**Survival by Sex**")
        fig6, ax6 = plt.subplots()
        sns.countplot(data=df, x='Sex', hue='Survived', ax=ax6)
        ax6.set_title("Survival Count by Sex")
        st.pyplot(fig6)

        st.markdown("**Survival by Pclass**")
        fig7, ax7 = plt.subplots()
        sns.countplot(data=df, x='Pclass', hue='Survived', ax=ax7)
        ax7.set_title("Survival Count by Pclass")
        st.pyplot(fig7)

        st.markdown("**Age vs Fare (colored by Survival)**")
        fig8, ax8 = plt.subplots()
        sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', ax=ax8)
        ax8.set_title("Age vs Fare Colored by Survival")
        st.pyplot(fig8)

        st.markdown("### Conclusion:")
        st.info("""
        - Females had a higher chance of survival.
        - Passengers from 1st class had better survival rates.
        - Younger and high-fare passengers showed better survival.
        """)