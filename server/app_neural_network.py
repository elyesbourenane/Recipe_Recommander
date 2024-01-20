from flask import Flask, request, jsonify
import nltk
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from flask_cors import CORS
import ast

app = Flask(__name__)
CORS(app)


max_words = 5000  # Maximum number of unique words to keep
max_len = 50  # Maximum length of a sequence

# Preprocess ingredients
nltk.download("stopwords")
stop_words = set(nltk.corpus.stopwords.words("english"))

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Load the model
model = tf.keras.models.load_model('recipe_recommendation_model.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load the dataset
df = pd.read_csv("recipes_3.csv")

# Function to recommend recipes based on input ingredients
def recommend_recipes(input_ingredients, n=10):
    processed_input = preprocess_text(input_ingredients)
    input_sequence = tokenizer.texts_to_sequences([processed_input])
    input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=max_len)
    
    prediction = model.predict(input_sequence)
    top_n_classes = np.argsort(prediction[0])[::-1][:n]
    
    # Convert predicted classes back to recipe titles
    recommended_recipes = [label_encoder.classes_[class_idx] for class_idx in top_n_classes]

    # df_temp = df[df["Title"].isin(recommended_recipes)]
    df_temp = df.query('Title in @recommended_recipes')
    maching_titles = df['Title'].apply(lambda x: titleEquals(x, recommended_recipes))
    # df_temp = df_temp[["Title", "Ingredients", "Instructions"]]
    
    return df[maching_titles][["Title", "Ingredients", "Instructions"]].to_dict(orient='records')

def titleEquals(title, list_title):
    for t in list_title:
        if title == t:
            return True
    return False

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
