import numpy as np
import pickle
import streamlit as st

#Loading the saved model
loaded_model = pickle.load(open('dia_trained_model.sav','rb'))

#Creating the function for prediction

def diabetes_prediction(input_data):
    # input_data = (11,130,76,0,0,33.2,0.42,56)
    # we need to change the data to numpy array
    input_data_as_numpyarray = np.asarray(input_data)
    # now we need to reshape this data
    input_data_reshaped = input_data_as_numpyarray.reshape(1,-1)

    print('input_data_reshaped :',input_data_reshaped)


    prediction = loaded_model.predict(input_data_reshaped)
    print('Final Prediction : ',prediction)

    if(prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'
    

st.sidebar.header('User input parameters')
def user_input():

    # giving a title
    st.title('Diabetes Prediction Web App')

    #getting the input from the users
    #Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    Pregnancies = st.sidebar.slider('No of Pregnancies',0,15,3)
    Glucose = st.sidebar.slider('Glucose level',40,350,120)
    BloodPressure = st.sidebar.slider('BloodPressure level',40,350,120)
    SkinThickness = st.sidebar.slider('SkinThickness ',0.000000,99.000000,20.536458)
    Insulin = st.sidebar.slider('Insulin level',0.000000,846.000000,79.799479)
    BMI = st.sidebar.slider('BMI',0.000000,67.100000,31.992578)
    DiabetesPedigreeFunction = st.sidebar.slider('DiabetesPedigreeFunction',0.078000,2.420000,0.471876)
    Age = st.sidebar.slider('Age',21,82,28)

    #code for prediction
    diagnosis = ''

    #Creating a button for prediction
    if st.button('Diabetes Test Results'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    if diagnosis == 'The person is diabetic':
        st.error(diagnosis)
    else:
        st.success(diagnosis)


if __name__ == '__main__':
    user_input()