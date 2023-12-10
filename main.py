from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
import unicodedata
import contractions
from nltk.tokenize import word_tokenize
import re
import nltk
from flask_cors import CORS

# Download 'stopwords' resource
nltk.download('stopwords')

# Utility Functions
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    return soup.get_text()

def remove_accented_chars(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def stopwords_removal(words):
    list_stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in words if word not in list_stopwords]

def pre_process_text(doc):
    doc = doc.lower()
    doc = strip_html_tags(doc)
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    doc = re.sub(r'[^a-zA-Z0-9\s%-]', '', doc)
    doc = re.sub(' +', ' ', doc).strip()
    tokens = word_tokenize(doc)
    return " ".join(stopwords_removal(tokens))

# Flask App Initialization
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load LSTM Model
modelLTSM = load_model('model.h5')  # Update with the actual filename

# Load Vectorizer
vectorizer = None  # Replace with your actual code to load the vectorizer

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=12000, lower=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data['text']
    processed_input = pre_process_text(user_input)

    user_input_sequence = tokenizer.texts_to_sequences([processed_input])
    user_input_padded = pad_sequences(user_input_sequence, maxlen=81, padding='post')

    y_predLTSM = modelLTSM.predict(user_input_padded)
    predicted_class_index = y_predLTSM.argmax(axis=1)[0]
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    ltsm_prediction = label_mapping[predicted_class_index]

    probabilities = modelLTSM.predict(user_input_padded)[0]

    percentages = [round(prob * 100, 2) for prob in probabilities]

    response = {
        "predictions": {
            "negative": f"{percentages[0]}%",
            "neutral": f"{percentages[1]}%",
            "positive": f"{percentages[2]}%"
        },
        "debug_info": {
            "processed_input": processed_input,
            "predicted_class": ltsm_prediction,
            "probabilities": {
                "negative": str(probabilities[0]),
                "neutral": str(probabilities[1]),
                "positive": str(probabilities[2])
            }
        }
    }

    return jsonify(response)

# Example Positive
example_text1 = "The company reported record-breaking profits this quarter, surpassing all analyst expectations. Investors are ecstatic about the robust financial performance, leading to a surge in stock prices."
# Example Neutral
example_text2 = "The Central Bank announced its decision to keep the interest rates unchanged, signaling stability in the financial markets."
# Example Negative
example_text3 = "The economic downturn has resulted in massive layoffs across various industries. Many businesses are struggling to stay afloat amidst declining consumer spending and market uncertainties. Investors are cautious about the market outlook."



if __name__ == '__main__':
    app.run(debug=True)
