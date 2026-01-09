
import streamlit as st
import pandas as pd
import pickle

# 🔹 1. Load Model and Preprocessing Pipeline

def load_model():
    with open("gbm_xgb_model.pkl", "rb") as f:
        return pickle.load(f)


# 🔹 2. User Input Function
def user_input():
    Age = st.slider("Age", 12, 19, 15)

    Sex_map = {"Male": 0, "Female": 1}
    Resident_map = {"Urban": 0, "Rural": 1}
    
    
    BMIg_map = {
        "Underweight": 0,
        "Normal weight": 1,
        "Overweight":2,
        "Obese": 2
    }

    Income_map = {
        "Low": 0,
        "Lower middle": 1,
        "Middle": 2,
        "High": 3
    }

    Smoking_map = {"Non-smoker": 0, "Smoker": 1}

    Alcohol_map = {
        "Non-drinker": 0,
        "1-2": 1,
        "3-5": 2,
        "6-9": 3,
        "≥10": 4
    }

    Stress_map = {"Mild": 0, "Moderate": 1, "High": 2, "Severe": 3}

    # 원문에 중복이 있어 3수준으로 정리(필요 시 라벨만 데이터셋 기준으로 변경)
    ParentsEdu_map = {
        "Middle school or lower": 0,
        "High school graduated": 1,
        "College or higher": 2
    }

    Edu_map = {"Low": 0, "Lower middle": 1, "Middle": 2, "Higher middle": 3, "High": 4}
    SRH_map = {"Low": 0, "Lower middle": 1, "Middle": 2, "Higher middle": 3, "High": 4}
    SPW_map = {"Low": 0, "Lower middle": 1, "Middle": 2, "Higher middle": 3, "High": 4}

    Sleep_map = {
        "Less than 4 hours": 0,
        "4 to 5 hours": 1,
        "5 to 6 hours": 2,
        "6 to 7 hours": 3,
        "7 to 8 hours": 4,
        "More than 8 hours": 5
    }

    YesNo_map = {"no": 0, "yes": 1}

    # 3) UI(선택) -> 숫자로 변환
    Sex = st.radio("Sex", list(Sex_map.keys()))
    Resident = st.radio("Resident (Region)", list(Resident_map.keys()))
    BMI_value  = st.number_input( "BMI", min_value=10.0, max_value=50.0, value=22.0,  step=0.1)
    
    BMI_g = st.selectbox("BMI group", list(BMIg_map.keys()))
    Income = st.selectbox("Income level", list(Income_map.keys()))

    Smoking = st.radio("Smoking", list(Smoking_map.keys()))
    Alcohol = st.selectbox("Alcohol consumption (drinks/week)", list(Alcohol_map.keys()))
    Stress = st.selectbox("Stress", list(Stress_map.keys()))

    Parents_education = st.selectbox("Parents education", list(ParentsEdu_map.keys()))
    Education = st.selectbox("Education", list(Edu_map.keys()))

    Self_reported_health = st.selectbox("Self-reported health", list(SRH_map.keys()))
    Self_perceived_weight = st.selectbox("Self-perceived weight", list(SPW_map.keys()))

    Sleep = st.selectbox("Sleep duration", list(Sleep_map.keys()))

    Substance_use = st.radio("Substance use", list(YesNo_map.keys()))
    Sexual_intercourse = st.radio("Sexual intercourse", list(YesNo_map.keys()))

    # 4) 최종 숫자 인코딩 값(모델 입력용)
    input_dict = {
        "Age": Age,
        "Sex": Sex_map[Sex],
        "Resident": Resident_map[Resident],   # 또는 컬럼명이 Region이면 "Region"으로 바꾸세요
        "BMI":BMI_value ,
        "BMI_g": BMIg_map[BMI_g],
        "Income": Income_map[Income],
        "Smoking": Smoking_map[Smoking],
        "Alcohol consumption": Alcohol_map[Alcohol],
        "Stress": Stress_map[Stress],
        "Parents education": ParentsEdu_map[Parents_education],
        "Education": Edu_map[Education],
        "Self-reported health": SRH_map[Self_reported_health],
        "Self-percieved weight": SPW_map[Self_perceived_weight],  # 철자(percieved) 데이터셋 컬럼과 맞추기
        "Sleep": Sleep_map[Sleep],
        "Substance use": YesNo_map[Substance_use],
        "Sexual intercourse": YesNo_map[Sexual_intercourse],
    }


    return pd.DataFrame([input_dict])

# 🔹 3. Prediction Function
def predict(input_df):
    model = load_model()
    proba = model.predict_proba(input_df)[0][1]  
    return proba

# 🔹 4. Streamlit App Main
def main():
    st.set_page_config(page_title="Adolescent Depression Risk Predictor", layout="centered")
    st.title("Adolescent Depression Risk Prediction App")
    st.markdown("Please enter your health check-up information below to estimate your information.")

    input_df = user_input()


    if st.button("Predict"):
        prob = predict(input_df)
        st.subheader("📈 Prediction Result")
        st.write(f"Predicted Risk Probability: **{prob:.2%}**")

        if prob >= 0.5:
            st.error("🔴 High risk detected. We recommend seeking medical consultation.")
        else:
            st.success("🟢 Your risk appears low. Maintain a healthy lifestyle!")

if __name__ == "__main__":
    main()
