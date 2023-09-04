 # import libraries
import streamlit as st
import pandas as pd
import joblib

# load the model pipeline object
model = joblib.load("model.joblib")

# add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

# deployment (Coding 1 Practical)

# age input form (numerical value)
age = st.number_input(label  = "01. Customer Age:",
                      min_value = 18,
                      max_value = 120,
                      value = 35)  # pre-populated value

# gender input form (categorical value)
gender = st.radio(label  = "02. Customer Gender:",
                  options = ["M", "F"]
                  )


# credit score input form
credit_score = st.number_input(label  = "03. Customer Credit Score:",
                      min_value = 0,
                      max_value = 1000,
                      value = 500) 


# submit inputs to model
if st.button("Submit for Predictions"):
    # store the data ina dataframe for prediction
    df = pd.DataFrame({"age": [age],
                       "gender": [gender],
                       "credit_score": [credit_score]})
    # apply model pipeline to the input data and
    # extract probability prediction
    pred_proba = model.predict_proba(df)[0][1]
    
    # output predictions
    st.subheader(f"Based on these customer attributes, the model predicts a purchase probability of  \
                 {pred_proba:.0%}")











