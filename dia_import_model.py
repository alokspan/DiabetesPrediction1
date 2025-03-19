import numpy as np
import pickle

#Loading the saved model
loaded_model = pickle.load(open('dia_trained_model.sav','rb'))


input_data = (11,130,76,0,0,33.2,0.42,56)
# we need to change the data to numpy array
input_data_as_numpyarray = np.asarray(input_data)
# now we need to reshape this data
input_data_reshaped = input_data_as_numpyarray.reshape(1,-1)

print('input_data_reshaped :',input_data_reshaped)


prediction = loaded_model.predict(input_data_reshaped)
print('Final Prediction : ',prediction)
