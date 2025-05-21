import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
hotel_details = pd.read_csv("F:\college work\FullstackHotelRecommendation\backend\Data\hotel_details.csv")
hotel_rooms = pd.read_csv("F:\Data\Modified_Hotel_Data.csv")
hotel_prices = pd.read_csv("F:\college work\FullstackHotelRecommendation\backend\Data\Modified_Hotel_data")  # New CSV for hotel prices

# Drop unnecessary columns in all datasets
for df in [hotel_details, hotel_rooms, hotel_prices]:
    df.drop(columns=[col for col in ['id', 'zipcode', 'latitude', 'longitude'] if col in df.columns], inplace=True)
    df.dropna(inplace=True)

hotel_details.drop_duplicates(subset='hotelid', keep=False, inplace=True)

# Merge datasets
hotel = pd.merge(hotel_rooms, hotel_details, left_on='hotelcode', right_on='hotelid', how='inner')
hotel = pd.merge(hotel, hotel_prices[['hotelcode', 'onsiterate']], on='hotelcode', how='left')  # Merge price data
hotel.drop(columns=['hotelid', 'curr', 'Source'], inplace=True)
hotel['description'] = hotel['ratedescription'] + hotel['roomamenities']
hotel.drop(columns=['roomamenities', 'ratedescription'], inplace=True)
hotel.drop_duplicates(inplace=True)
hotel.dropna(inplace=True)

# Add guest number using room types
def calc_guests(hotel_df):
    room_no = [
        ('king', 2), ('queen', 2), ('triple', 3), ('master', 3), ('family', 4),
        ('murphy', 2), ('quad', 4), ('double-double', 4), ('mini', 2), ('studio', 1),
        ('junior', 2), ('apartment', 4), ('double', 2), ('twin', 2), ('double-twin', 4),
        ('single', 1), ('disabled', 1), ('accessible', 1), ('suite', 2), ('one', 2)
    ]
    guests_no = []
    for roomtype in hotel_df['roomtype']:
        roomtype = str(roomtype).lower()
        for word in roomtype.split():
            for room, guests in room_no:
                if word == room:
                    guests_no.append(guests)
                    break
            else:
                continue
            break
        else:
            guests_no.append(2)
    return guests_no

hotel['guests_no'] = calc_guests(hotel)
hotel['description'] = hotel['description'].str.replace(r': ;', ',', regex=True)

# Define Recommender Class
class HotelRecommender:
    def __init__(self, hotel_df):
        self.hotel = hotel_df.copy()
        self.hotel['city'] = self.hotel['city'].str.lower()
        self.hotel['description'] = self.hotel['description'].str.lower()
        self.stop_words = set(stopwords.words('english'))
        self.lemm = WordNetLemmatizer()

    def citybased(self, city):
        city = city.lower()
        df = self.hotel[self.hotel['city'] == city]
        df = df.sort_values(by='starrating', ascending=False).drop_duplicates(subset='hotelcode')
        return df[['hotelname', 'price', 'guests_no', 'starrating', 'description', 'url']].head(10)

    def pop_citybased(self, city, number):
        city = city.lower()
        df = self.hotel[(self.hotel['city'] == city) & (self.hotel['guests_no'] == number)]
        df = df.sort_values(by='starrating', ascending=False).drop_duplicates(subset='hotelcode')
        return df[['hotelname', 'roomtype', 'guests_no', 'starrating', 'address']].head(10)

    def requirementbased(self, city, number, features):
        city = city.lower()
        features = features.lower()
        features_tokens = word_tokenize(features)
        f_set = {self.lemm.lemmatize(w) for w in features_tokens if w not in self.stop_words}

        df = self.hotel[(self.hotel['city'] == city) & (self.hotel['guests_no'] == number)].copy()
        df = df.sort_values(by='starrating', ascending=False).reset_index(drop=True)

        similarity = []
        for desc in df['description']:
            tokens = word_tokenize(desc)
            temp_set = {self.lemm.lemmatize(w) for w in tokens if w not in self.stop_words}
            similarity.append(len(temp_set.intersection(f_set)))

        df['similarity'] = similarity
        df = df.sort_values(by='similarity', ascending=False).drop_duplicates(subset='hotelcode')

        return df[['hotelname', 'roomtype', 'price', 'guests_no', 'starrating',
                   'address', 'similarity', 'description', 'url']].head(10)

    def budget_based_recommendation(self, city, number, budget_min, budget_max):
        city = city.lower()
        df = self.hotel[(self.hotel['city'] == city) & (self.hotel['guests_no'] == number)]
        df = df[(df['price'] >= budget_min) & (df['price'] <= budget_max)]  # Filter based on budget
        df = df.sort_values(by='starrating', ascending=False).drop_duplicates(subset='hotelcode')
        return df[['hotelname', 'roomtype', 'price', 'guests_no', 'starrating', 'address']].head(10)

# Create and save model
model = HotelRecommender(hotel)

with open('hotel_recommender_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as 'hotel_recommender_model.pkl'")