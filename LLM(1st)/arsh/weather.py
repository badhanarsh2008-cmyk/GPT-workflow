import requests

API_KEY = "c0f36190065042e7a7283730260703"

city = input("Enter city name: ")

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

response = requests.get(url)
data = response.json()

# check if request worked
if response.status_code == 200:

    temperature = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    weather = data["weather"][0]["description"]

    print("City:", city)
    print("Temperature:", temperature, "°C")
    print("Humidity:", humidity)
    print("Weather:", weather)

else:
    print("Error:", data["message"])