import streamlit as st
import joblib
import numpy as np

st.title('Bank Loan Approval with Machine Learning')
model = joblib.load('pickles/RandomForest.pkl')
encodeMap ={
    "female": 0,
    "male": 1,
    "Associate": 0,
    "Bachelor": 1,
    "Doctorate": 2,
    "High School": 3,
    "Master": 4,
    "MORTGAGE": 0,
    "OTHER": 1,
    "OWN": 2,
    "RENT": 3,
    "DEBTCONSOLIDATION": 0,
    "EDUCATION": 1,
    "HOMEIMPROVEMENT": 2,
    "MEDICAL": 3,
    "PERSONAL": 4,
    "VENTURE": 5,
    "No": 0,
    "Yes": 1
}

def scale(toBeScaled, filename):
    filepath = f'pickles/{filename}'
    scaler = joblib.load(filepath)
    toBeScaled = scaler.transform(np.array([[toBeScaled]]))
    return toBeScaled[0][0]

def makePrediction(features):
    params = np.array(features).reshape(1, -1)
    prediction = model.predict(params)
    return prediction[0]

def main():
    personAge = st.number_input(label='Insert age:', min_value=18, max_value=50, step=1)
    personGender = st.pills(label='Insert gender:', options=['male', 'female'])
    personEducation = st.selectbox(label='Insert education:', options=['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
    personIncome = st.number_input(label='Insert income($) in a year:', min_value=0)
    personEmpExp = st.slider(label='Insert Person\'s Employment Experience:', min_value=0, max_value=50, step=1)
    personHomeOwnership = st.pills(label='Insert home ownership:', options=['RENT', 'MORTGAGE', 'OWN', 'OTHER'])
    loanAmount = st.number_input(label='Insert loan amount:', min_value=1000, max_value=200000)
    loanIntent = st.selectbox(label='Insert loan intention:', options=['DEBTCONSOLIDATION','EDUCATION','HOMEIMPROVEMENT','MEDICAL', 'PERSONAL', 'VENTURE'])
    loanIntRate = st.slider(label='Insert loan interest rate:', min_value=5.0, max_value=25.0, step=0.1)
    loanPercentIncome = np.round(personIncome/loanAmount, 2)
    cbPersonCredHistLength = st.number_input(label='Insert loan duration in year:', min_value = 1, max_value=20)
    creditScore = st.number_input(label='Insert credit score:', min_value=0)
    previousLoanDefaultsOnFile = st.selectbox(label='Insert previous loan default status:', options=['Yes', 'No'])
    
    if st.button('Make Prediction'):
        personGender = encodeMap[personGender]
        personEducation = encodeMap[personEducation]
        personHomeOwnership = encodeMap[personHomeOwnership]
        loanIntent = encodeMap[loanIntent]
        previousLoanDefaultsOnFile = encodeMap[previousLoanDefaultsOnFile]
        
        personAge = scale(personAge, 'person_ageScaler.pkl')
        personIncome = scale(personIncome, 'person_incomeScaler.pkl')
        personEmpExp = scale(personEmpExp, 'person_emp_expScaler.pkl')
        loanAmount = scale(loanAmount, 'loan_amntScaler.pkl')
        loanIntRate = scale(loanIntRate, 'loan_int_rateScaler.pkl')
        loanPercentIncome = scale(loanPercentIncome, 'loan_percent_incomeScaler.pkl')
        cbPersonCredHistLength = scale(cbPersonCredHistLength, 'cb_person_cred_hist_lengthScaler.pkl')
        creditScore = scale(creditScore, 'credit_scoreScaler.pkl')

        features = [personAge, personGender, personEducation, personIncome, personEmpExp, personHomeOwnership,
                    loanAmount, loanIntent, loanIntRate, loanPercentIncome, cbPersonCredHistLength, creditScore, previousLoanDefaultsOnFile]
        result = makePrediction(features)
        if result==1:
            result = 'Approved'
            st.success(f'Your proposal is {result}')
        elif result==0:
            result = 'Rejected'
            st.error(f'Your proposal is {result}')
        else:
            st.error(f'Machine is not available at the moment')

if __name__ == '__main__':
    main()