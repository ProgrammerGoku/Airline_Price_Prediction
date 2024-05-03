import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
with open('gradient_boost_reg.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl','rb') as file:
     scale = pickle.load(file)
# Define function to make predictions
def predict(input_data):
    # Convert input_data into DataFrame
    data = pd.DataFrame(input_data, index=[0])
    data = scale.transform(data)
    # Make prediction
    prediction = model.predict(data)
    return prediction

# Streamlit app
def main():
    st.title('Airline Price Prediction App')    
    # Input fields
    input_columns = [
        "Total_Stops", "Duration_mins", "Date", "Month", "Airline_LabelEncoded",
        "Source_LabelEncoded", "Destination_LabelEncoded", "Duration_LabelEncoded",
        "Additional_Info_LabelEncoded", "departure_slot", "arrival_slot",
        "day_of_departure", "day_of_arrival", "day_type"
    ]    
    input_data = {}
    airlines=['Air Asia','Air India','GoAir','Jet Airways','Jet Airways Business','Multiple carriers', 
         'Multiple carriers Premium economy','SpiceJet','Trujet','Vistara','Vistara Premium economy', 
       ]
    sources=['Banglore','Chennai','Delhi','Kolkata','Mumbai']
    destinations=[ 'Banglore','Cochin','Delhi','Hyderabad','Kolkata', ]
    infos=['No info', 'In-flight meal not included',
       'No check-in baggage included', '1 Short layover', 'No Info',
       '1 Long layover', 'Change airports', 'Business class',
       'Red-eye flight', '2 Long layover']
    dept_slots=['6PM to 12AM', '12AM to 6AM', '6AM to 12PM', '12 PM to 6PM']
    arrival_slots=['12AM to 6AM', '12 PM to 6PM', '6PM to 12AM', '6AM to 12PM']
    dept_day=['Friday','Monday', 'Saturday','Sunday','Thursday','Tuesday','Wednesday']

    input_data['Total_Stops']=st.number_input("Total Stops",min_value=1,format="%d") #Get Total Stops

    st.write("Duration")
    hours = st.slider("Hours", 0, 23, 0)
    minutes = st.slider("Minutes", 1, 59, 1)

    input_data['Duration_mins']=hours*60+minutes #Convert to duration to mins

    selected_date = st.date_input("Select a date")

    input_data['Date']=selected_date.day #Get the day from date
    input_data['Month']=selected_date.month #Get the month from date

    airline_option=st.selectbox('Select Airline',airlines)

    input_data["Airline_LabelEncoded"]=airlines.index(airline_option) #Get the index of airline chose

    source_option=st.selectbox('Select Source point',sources)

    input_data['Source_LabelEncoded']=sources.index(source_option) #Get the index of source chose

    destinations=[dest for dest in destinations if dest!=source_option]

    destination_option=st.selectbox("Select Destination",destinations) #Chosen source is not displayed in destination

    input_data["Destination_LabelEncoded"]=destinations.index(destination_option)

    input_data['Duration_LabelEncoded']=st.number_input("Duration Label",min_value=0,format="%d")

    infos_option=st.selectbox("Select Additionals",infos)
    input_data["Additional_Info_LabelEncoded"]=infos.index(infos_option) #Get the index of additional info

    dept_slot_option=st.selectbox("Select Departure Time Slot",dept_slots)
    input_data["departure_slot"]=dept_slots.index(dept_slot_option) #Get the index of dept_slot

    arrival_slot_option=st.selectbox("Select Arrival Time Slot",arrival_slots)
    input_data['arrival_slot']=arrival_slots.index(arrival_slot_option) #Get the index of arrival_slot

    input_data["day_of_departure"]= dept_day.index(selected_date.strftime("%A")) #Get the day of departure

    input_data["day_of_arrival"]= dept_day.index(selected_date.strftime("%A")) #Get the day of arrival

    input_data['day_type']= 1 if selected_date.strftime("%A") in ['Saturday','Sunday'] else 0 #Get the day type

    # Display bar graph for prices of other airlines
    if airline_option:
        prices = [100, 200, 150, 300, 250, 180, 210, 220, 190, 280]  # Example prices for demonstration
        plt.bar(airlines, prices)
        plt.xlabel('Airlines')
        plt.ylabel('Prices')
        plt.title('Prices of Other Airlines')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    if st.button('Predict'):
        # Make prediction
        prediction = predict(input_data)
        st.success(f'The predicted price is: {prediction[0]:.2f}')

if __name__ == '__main__':
    main()
