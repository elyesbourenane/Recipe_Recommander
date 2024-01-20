from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS
import ast


app = Flask(__name__)
CORS(app)

# Load data
df = pd.read_csv("recipes.csv")

# Preprocessing of ingredients
stop_words = set(nltk.corpus.stopwords.words("english"))

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Apply the preprocessing
df["Processed_Ingredients"] = df["Ingredients"].apply(preprocess_text)

# Calculate TF-IDF Vectors for Ingredients
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Processed_Ingredients'].tolist())

def recommend_recipes(input_ingredients, n=10):
    processed_input = preprocess_text(input_ingredients)
    input_vector = tfidf_vectorizer.transform([processed_input])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]
    recommended_recipes = df.iloc[sorted_indices[:n]].drop("Processed_Ingredients", axis=1)
    recommended_recipes = recommended_recipes[["Title", "Ingredients", "Instructions"]]
    recommended_dict = recommended_recipes.to_dict(orient='records')
    return recommended_dict

@app.route('/api/recommend-recipes', methods=['POST'])
def recommend_api():
    input_ingredients = request.json['ingredients']
    recommendations = recommend_recipes(input_ingredients)
    recommendations_modified = []
    for recipe in recommendations:
        recipe_tmp = recipe
        recipe_tmp["Ingredients"] = ast.literal_eval(recipe_tmp["Ingredients"])
        if len(recipe_tmp["Ingredients"]) > 8:
            recipe_tmp["Ingredients"] = recipe_tmp["Ingredients"][:8]

        recipe_tmp["Instructions"] = recipe_tmp["Instructions"].split(".")[:-1]

        recommendations_modified.append(recipe_tmp)
    return jsonify(recommendations_modified)

if __name__ == "__main__":
    app.run(debug=True)