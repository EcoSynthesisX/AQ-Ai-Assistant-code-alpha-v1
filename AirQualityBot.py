# Importing the necessary libraries
import requests
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import math
import os
from dotenv import load_dotenv
load_dotenv()

# Loading API key from environment variables
API_KEY_OPEN_WEATHER = os.environ.get('OPEN_WEATHER_API_KEY', 'default_value')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'default_value')

# Checking if CSV File Exists
csv_file_path = 'Air Quality Table - LandR.csv'
if os.path.exists(csv_file_path):
    combined_levels_and_rec_df = pd.read_csv(csv_file_path)
else:
    print(f"Error: File {csv_file_path} not found.")
    # Handle the error as needed, for example, by exiting the program

# Set up API and locations
API_BASE_URL_OPEN_WEATHER = "http://api.openweathermap.org/data/2.5/air_pollution"
lat = 9.706497174
lon = 99.985496058

def get_current_air_pollution_data():
    url = f"{API_BASE_URL_OPEN_WEATHER}?lat={lat}&lon={lon}&appid={API_KEY_OPEN_WEATHER}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx and 5xx)
        
        data = response.json()
        
        if 'coord' in data and 'list' in data:
            return data
        else:
            print(f"Error fetching data: {data}")
            return None
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

current_air_pollution_data = get_current_air_pollution_data()

def get_pollutant_levels_and_recommendations(api_response, combined_df):
    """
    Function to match pollutant values from API response with air quality levels and recommendations.
    
    Parameters:
    api_response (dict): Dictionary containing the entire API response.
    combined_df (DataFrame): DataFrame containing the combined air quality levels and recommendations table.
    
    Returns:
    dict: A dictionary containing the qualitative name, index, and recommendation for each pollutant, 
          along with the timestamp from the API response.
    """
    # Extract the list of air quality data from the API response
    air_quality_data_list = api_response['list']
    
    # Initialize an empty list to store the levels and recommendations for each data point in the API response
    results = []
    
    # Map API pollutant names to DataFrame column names
    pollutant_column_map = {
        'co': 'CO',
        'no2': 'NO2',
        'o3': 'O3',
        'so2': 'SO2',
        'pm2_5': 'PM2.5',
        'pm10': 'PM10'
    }
    
    # Iterate through each data point in the API response
    for data_point in air_quality_data_list:
        # Extract the pollutant concentrations and timestamp from the data point
        components = data_point['components']
        timestamp = data_point['dt']
        
        # Initialize an empty dictionary to store the levels and recommendations for this data point
        pollutant_info = {'Timestamp': timestamp}
        
        # Iterate through each pollutant in the API response data
        for api_pollutant, df_pollutant in pollutant_column_map.items():
            # Get the value of the pollutant from the components data
            value = components[api_pollutant]
            
            # Find the corresponding row in the combined_df DataFrame
            level_and_rec_row = combined_df[
                (combined_df[f'{df_pollutant} Lower'] <= value) & 
                (combined_df[f'{df_pollutant} Upper'] >= value)
            ].iloc[0]
            
            # Store the qualitative name, index, and recommendation for this pollutant in the dictionary
            pollutant_info[api_pollutant] = {
                'Qualitative Name': level_and_rec_row['Qualitative Name'],
                'Index': level_and_rec_row['Index'],
                'Recommendation': level_and_rec_row[f'{df_pollutant} Recommendations']
            }
        
        # Add the pollutant info for this data point to the results list
        results.append(pollutant_info)
    
    return results

# Store the result of the function in a variable
pollution_levels_and_recommendations = get_pollutant_levels_and_recommendations(current_air_pollution_data, combined_levels_and_rec_df)

def generate_message(pollution_levels_and_recommendations):
    # Convert timestamp to local time
    timestamp = pollution_levels_and_recommendations[0]['Timestamp']
    local_time = datetime.fromtimestamp(timestamp)
    time_str = local_time.strftime('%H:%M %p on %d %B %Y')
    
    # Determine overall pollution level
    max_index = -1
    pollutant_levels = {}  # To store pollutant levels separately

    for pollutant, data in pollution_levels_and_recommendations[0].items():
        if pollutant != 'Timestamp':
            pollutant_levels[pollutant] = data['Index']  # Extracting the Index value
            if data['Index'] > max_index:
                max_index = data['Index']
                max_qualitative_name = data['Qualitative Name']
    
    # Extract and combine recommendations into bullet points
    recommendations_list = [f"- {data['Recommendation']}" for pollutant, data in pollution_levels_and_recommendations[0].items() if pollutant != 'Timestamp' and (not isinstance(data['Recommendation'], float) or (isinstance(data['Recommendation'], float) and not math.isnan(data['Recommendation'])))]

    # Generate the final message with new formatting
    greetings = f"Citizens of Koh-Phangan!\nNow it is {time_str}, air condition is {max_qualitative_name.lower()}."
    recommendations = '\n'.join(recommendations_list)
    pollutant_levels = f"{pollutant_levels}"
    
    return greetings, recommendations, pollutant_levels


# Generate the message based on the current air pollution data
greetings, recommendations, pollutant_levels = generate_message(pollution_levels_and_recommendations)

#Model version
GPT3 = 'gpt-3.5-turbo-16k-0613'
GPT4 = 'gpt-4-0613'

chat_greetings_template = PromptTemplate(
    input_variables=['greetings'],
    template="""
            You are an expert greeting people. Your task is to welcome people who are reading this message 
            with warm welcome. Use emojis in your response. 
            Say good day, good night, depending on a time. Inform about current date and time. Max message lenght is 30 words
            Current conditions are {greetings}
    """
)

recommendations_template = PromptTemplate(
    input_variables=['recommendations'],
    template="""
            You are an expert in providing concise recommendations based on current air pollution levels 
            and recommendations. Your goal is to provide recommendations data that you will receive in concise manner 
            keeping only meaning with minimal amount of text removing any repetetivness
            Start with: Based on air pollution recommendations are:
            Current recommmendations are {recommendations}
    """
)

greet_llm = ChatOpenAI(temperature=1, model = GPT4, openai_api_key=OPENAI_API_KEY)
recommendations_llm = ChatOpenAI(temperature=0, model = GPT4, openai_api_key=OPENAI_API_KEY)

chat_greetings_chain = LLMChain(llm=greet_llm, prompt=chat_greetings_template, output_key='chat_greetinigs')
chat_recommendations_chain = LLMChain(llm=recommendations_llm, prompt=recommendations_template, output_key='chat_recommendations')

chat_greetings = chat_greetings_chain.run(greetings=greetings)
chat_recommendations = chat_recommendations_chain.run(recommendations=recommendations)

print(chat_greetings)
print(chat_recommendations)

AirQualityBot = ChatOpenAI(temperature=0.1, model = GPT4)

history = ChatMessageHistory()

start_message = f"{chat_greetings} {chat_recommendations}"

role = """

Purpose:
The AirQuality bot serves as an informational assistant providing current air quality updates and health 
recommendations based on chat history. 

Health Recommendations: Offer health-related guidance tailored to the current air 
quality conditions, specifically targeting sensitive groups when the AQI is at levels that warrant caution.

Precautionary Advice: When the AQI enters ranges considered unhealthy for sensitive groups
 or worse, include clear instructions on how to minimize health risks.

Behavioral Do's:

Deliver messages in a calm and informative tone.
Use language that is easily understood by non-experts.
Reference reputable sources for data and recommendations.
Update dynamically according to the latest air quality readings.

Behavioral Don'ts:
Don't ask user for location, it is provided in AI history messages
Do not provide medical advice beyond general recommendations for air quality.
Avoid alarming language that may cause unnecessary panic.
Do not speculate about future air quality conditions.
Refrain from using technical jargon without explanations.

User Engagement:

Encourage users to ask questions about air quality.
Provide tips on how to stay informed and protect oneself against air pollution.

Safery Consideration:
The bot should not perform functions beyond providing air quality updates and
associated health recommendations.
"""

history.add_ai_message(role)
history.add_ai_message(start_message)

print(history.messages)


# Start the conversation loop.
while True:
    # Get user input.
    user_input = input("User: ")

    # Check if the user wants to end the conversation.
    if user_input.lower() == 'quit':
        break

    # Add user message to the history.
    history.add_user_message(user_input)

    # Get AI response
    ai_response = str(AirQualityBot(history.messages))

    # Add AI message to the history.
    history.add_ai_message(ai_response)

    # Display the AI response.
    print("AI:", ai_response)
