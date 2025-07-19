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
img = Image.open('titanic-ship.png')
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

        st.markdown("**Survival by Gender**")
        fig_gender_survival, ax = plt.subplots(figsize=(10, 5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        ax = sns.countplot(x='Survived', data=df, hue='Sex', ax=ax)
        for bars in ax.containers:
            ax.bar_label(bars)

        plt.xticks([0, 1], ['Not Survived', 'Survived'])
        ax.set_title('Survival by Gender')
        ax.legend(title='Sex')
        st.pyplot(fig_gender_survival)


        st.markdown("**Survival by Passenger Class (Pclass)**")
        fig_pclass_survival, ax = plt.subplots(figsize=(10, 5))
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        ax = sns.countplot(data=df, x='Pclass', hue='Survived', ax=ax)
        for bars in ax.containers:
            ax.bar_label(bars)

        ax.set_title("Survival Count by Passenger Class")
        ax.legend(title='Survived', labels=['Not Survived', 'Survived'])
        st.pyplot(fig_pclass_survival)


        # Age Distribution KDE Plot
        st.markdown("**Age Distribution by Survival**")
        fig_kde, ax_kde = plt.subplots(figsize=(10, 5))
        ax_kde.grid(axis='y', linestyle='--', alpha=0.7)

        sns.kdeplot(df[df['Survived'] == 1]['Age'], label='Survived', fill=True, ax=ax_kde)
        sns.kdeplot(df[df['Survived'] == 0]['Age'], label='Did Not Survive', fill=True, ax=ax_kde)

        ax_kde.set_title("Age Distribution by Survival")
        ax_kde.set_xlabel("Age")
        ax_kde.set_ylabel("Density")
        ax_kde.legend()
        st.pyplot(fig_kde)

        # Conclusion
        st.markdown("### Conclusion of age-distribution :")
        st.markdown("""
        - Females had a higher chance of survival.
        - Passengers from 1st class had better survival rates.
        - Younger and high-fare passengers showed better survival.
        - **From the age distribution plot**, passengers who did not survive were more concentrated in the 20–40 age range, while survivors included more children (ages 0–10) and some older adults. This supports the idea that rescue protocols like *'women and children first'* played a key role in survival outcomes.
        """)

        st.markdown("**Age vs Fare (colored by Survival)**")
        fig8, ax8 = plt.subplots()
        sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', ax=ax8)
        ax8.set_title("Age vs Fare Colored by Survival")
        st.pyplot(fig8)


        st.markdown("### Final Conclusion from Bivariate Analysis")

        st.markdown("""
        #### **Gender vs Survival**
        - **Females exhibited significantly higher survival rates compared to males.**  
        This outcome aligns with historical accounts and rescue priorities during the Titanic disaster, where *"women and children first"* was a guiding principle during evacuation efforts.

        #### **Passenger Class (Pclass) vs Survival**
        - **Passengers traveling in First Class (Pclass = 1) had the highest survival rate**, while those in Third Class experienced the lowest.  
        This indicates a strong socio-economic bias in survival, likely influenced by cabin proximity to lifeboats, access to crew guidance, and prioritization during rescue.

        #### **Age vs Survival**
        - **Survivors tended to include more young children (ages 0–10) and some older adults**, while **non-survivors were more concentrated in the 20–40 age range**.  
        This supports the theory that families and younger passengers received preferential treatment during rescue operations.

        #### **Fare vs Age (Colored by Survival)**
        - **Passengers who paid higher fares (mostly First Class) and younger passengers were more likely to survive.**  
        The scatter plot highlights a survival cluster among wealthier and younger individuals, indicating a combined effect of socio-economic status and age on survival outcomes.
        """)

    st.markdown(
        """
        <div style="text-align: center; padding: 10px; font-size: 14px;">
            Made with <span style="color: #e25555;">❤️</span> by <strong>Faisal Khan</strong>
        </div>
        """,
        unsafe_allow_html=True
    )
