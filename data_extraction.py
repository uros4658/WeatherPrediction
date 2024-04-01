import requests
import pandas as pd
from datetime import datetime, timedelta

# Your API Key
api_key = "4e45ef1900365aa87ed1695a28ac9c41"

# The location you're interested in
latitude = 44.7866
longitude = 20.4489

# Define the start and end dates for the historical period you're interested in
end_date = datetime.now()
start_date = end_date - timedelta(days=50)

# Initialize a list to hold all the data
all_data = []

# Loop over each day in the range
for day in range((end_date - start_date).days):
    # Convert the current day to a date
    current_date = start_date + timedelta(days=day)

    # Convert the current date to a UNIX timestamp
    current_timestamp = int(current_date.timestamp())

    # Make a request to the API
    response = requests.get(f"https://history.openweathermap.org/data/2.5/history/city?lat={latitude}&lon={longitude}&type=hour&start={current_timestamp}&end={current_timestamp + 86400}&appid={api_key}")

    # Convert the response to JSON
    data = response.json()

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the data you're interested in
        for hour_data in data['list']:
            temperature = hour_data['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
            humidity = hour_data['main']['humidity']
            wind_speed = hour_data['wind']['speed']
            weather_conditions = hour_data['weather'][0]['description']

            # Add the data to the list
            all_data.append([temperature, humidity, wind_speed, weather_conditions])
    else:
        print(f"Error: {data['message']}")

# Create a DataFrame from the data
df = pd.DataFrame(all_data, columns=['Temperature', 'Humidity', 'Wind Speed', 'Weather Conditions'])

# Save the DataFrame to a CSV file
df.to_csv("weather_data.csv")
