# Hotel-recommendation-system

🏨 Hotel Recommendation System
This project builds a Hotel Recommendation Model that suggests hotels to users based on city, room capacity, preferences (keywords), and budget constraints. It uses NLP techniques to parse and compare user requirements with hotel descriptions. The final model is serialized as a pickle file for easy integration into web applications.

📊 Data Files Required
Ensure these files exist and the paths are correctly set in hotel_model.py:
hotel_details.csv
Modified_Hotel_Data.csv

🧠 Features of the Model
The HotelRecommender class provides:

1. citybased(city)
Returns top 10 hotels in a city based on star rating.
2. pop_citybased(city, number)
Filters hotels in a city that accommodate a specific number of guests.
3. requirementbased(city, number, features)
Recommends hotels based on user-described features using NLP similarity.
4. budget_based_recommendation(city, number, budget_min, budget_max)
Suggests hotels within a given price range.

