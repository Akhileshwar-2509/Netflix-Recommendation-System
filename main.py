import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("netflixData.csv")
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Keep only relevant columns
data = data[["Title", "Description", "Content Type", "Genres"]]
print(data.head())

# Drop rows with missing values
data = data.dropna()

import nltk
import re
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string

# Load the set of stopwords
stopword = set(stopwords.words('english'))

# Function to clean text data
def clean(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub('\[.*?\]', '', text)  # Remove text within brackets
    text = re.sub('https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub('<.*?>+', '', text)  # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub('\n', '', text)  # Remove newlines
    text = re.sub('\w*\d\w*', '', text)  # Remove words containing numbers
    text = [word for word in text.split(' ') if word not in stopword]  # Remove stopwords
    text = " ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]  # Apply stemming
    text = " ".join(text)
    return text

# Clean the 'Title' column
data["Title"] = data["Title"].apply(clean)
print(data.Title.sample(10))

# Convert the 'Genres' column into a list of strings for vectorization
feature = data["Genres"].tolist()

# Apply TF-IDF vectorization on the 'Genres' feature
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)

# Compute the cosine similarity matrix
similarity = cosine_similarity(tfidf_matrix)

# Create a Series mapping each title to its index
indices = pd.Series(data.index, index=data['Title']).drop_duplicates()

# Function to get Netflix recommendations based on the title
def netFlix_recommendation(title, similarity=similarity):
    index = indices[title]  # Get the index of the title
    similarity_scores = list(enumerate(similarity[index]))  # Get similarity scores for all titles
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)  # Sort by similarity score
    similarity_scores = similarity_scores[1:11]  # Get the top 10 most similar titles (excluding the title itself)
    movieindices = [i[0] for i in similarity_scores]  # Get the indices of these titles
    return data['Title'].iloc[movieindices]  # Return the titles of the recommended movies/shows

# Test the recommendation system with an example title
print("Enter example title")
text=input()
print(netFlix_recommendation(text))
