import streamlit as st
import joblib
import numpy as np

st.title('Bank Loan Approval with Machine Learning')
model = joblib.load('RandomForest.pkl')

def main():
    st.title('Machine Learning Model Deployment')

    personAge = st.number_input(label='Insert age:', min_value=18, max_value=50, step=1)
    personGender = st.pills(label='Insert gender:', options=['male', 'female'])
    personEducation = st.selectbox(label='Insert education:', options=['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    personIncome = st.number_input(label='Insert income($) in a year:', min_value=0)
    personEmpExp = st.slider(label='Insert Person\'s Employment Experience:', min_value=0, max_value=50, step=1)
    personHomeOwnership = st.pills(label='Insert education:', options=['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    loanAmount = st.number_input(label='Insert loan amount:', min_value=1000, max_value=200000)
    loanIntent = st.selectbox(label='Insert loan intention:', options=['DEBTCONSOLIDATION','EDUCATION','HOMEIMPROVEMENT','MEDICAL', 'PERSONAL', 'VENTURE'])
    loanIntRate = st.slider(label='Insert loan interest rate:', min_value=5, max_value=25, step=0.1)
    loanPercentIncome = np.round(personIncome/loanAmount, 2)
    cbPersonCredHistLength = st.number_input(label='Insert loan duration in year:', min_value = 1, max_value=20)
    creditScore = st.number_input(label='Insert credit score:', min_value=0)
    previousLoanDefaultsOnFile = st.pills(label='Insert education:', options=['Yes', 'No'])
    
    if st.button('Make Prediction'):
        features = [personAge, personGender, personEducation, personIncome, personEmpExp, personHomeOwnership,
                    loanAmount, loanIntent, loanIntRate, loanPercentIncome, cbPersonCredHistLength, creditScore, previousLoanDefaultsOnFile]
        result = makePrediction(features)
        if result==1:
            result = 'Approved'
        else:
            result = 'Rejected'
        st.success(f'Your proposal is {result}')

def makePrediction(features):
    params = np.array(features).reshape(1, -1)
    prediction = model.predict(params)
    return prediction[0]

if __name__ == '__main__':
    main()
