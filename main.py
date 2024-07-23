import joblib
import streamlit as st
import re
import pandas as pd
from PIL import Image
# from keras.src.utils import pad_sequences
from keras.preprocessing.sequence import pad_sequences
from pyvi import ViUtils
import nltk
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from keras_preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
import random
import requests
from bs4 import BeautifulSoup

def fetch_website_content(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
    except:
        return None

# Function to inject ads into website content
def inject_ads(html_content):
    # Modify this function to inject ads into the HTML content
    # For simplicity, this example just adds a placeholder div with ad code
    soup = BeautifulSoup(html_content, 'html.parser')
    ad_div = soup.new_tag('div')
    ad_div.string = 'Ad code goes here'
    ad_div['style'] = 'height: 200px; margin-bottom: 20px;'
    # Insert the ad div into the appropriate location in the HTML content
    # (replace this with your logic to identify where to place ads)
    main_content_div = soup.find('div', id='main-content')
    if main_content_div:
        main_content_div.insert(0, ad_div)
    return str(soup)

names=['URL','Category']
df=pd.read_csv('full training data.csv',names=names, usecols=[0, 2], na_filter=False,  encoding='latin-1')

# stem word for svm
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english', ngram_range=(3, 3))

categories = ['Automotive', 'Books & Literature', 'Business & Finance',
       'Careers', 'Education', 'Entertainment & Art',
       'Family & Relationships', 'Food & Drink', 'Healthy Living',
       'Home & Garden', 'News & Politics', 'Real Estate', 'Science & Technology',
       'Sports', 'Style & Fashion', 'Travel', 'Games', 'Laws & Policies', 'Environment' ]

gif_paths = {
    'Automotive': ['media/automotive/honda.jpg'],
    'Books & Literature': ['media/book&literature/book1gif'],
    'Business & Finance': ['media/business/business1.gif'],
    'Careers': ['media/career/career1.gif'],
    'Education': ['media/career/edu1.jpg'],
    'Entertainment & Art': ['media/entertainment/demo1.jpg'],
    'Family & Relationships': ['media/family/family1.gif'],
    'Food & Drink': ['media/food/food1.jpg'],
    'Healthy Living': ['media/healthy/healthy1.jpg'],
    'Home & Garden': ['media/home/home1.gif'],
    'News & Politics': ['media/news/news1.gif'],
    'Real Estate': ['media/real/real1.gif'],
    'Science & Technology': ['media/science/science1.gif'],
    'Sports': ['media/sport/sport1.jpg'],
    'Style & Fashion': ['media/style/style1.gif'],
    'Travel': ['media/travel/travel1.jpg', 'media/travel/travel2.gif'],
    'Games': ['media/game/game1.gif'],
    'Laws & Policies': ['media/law/law1.gif'],
    'Environment': ['media/environment/environment1.gif'],
}

xgb_model = joblib.load('xgboost_1.4.1.pkl')
rf_model = joblib.load('random_forest_1.4.1.pkl')
dt_model = joblib.load('decision_tree_1.4.1.pkl')
logis_model = joblib.load('logistic_regression.pkl')
cnn_model = load_model('cnn_hyper_2_16.h5')
svm_model = joblib.load('origin_svm_1.4.1.pkl')

tokenizer = Tokenizer(oov_token='<OOV>')
def url_to_text(url):
  # remove stopwords
  url = url.replace('.html','').replace('.htm','').replace('http://','').replace('https://','')
  url = re.sub('^(.*?/)','/', url) # remove domains
  url = re.sub('[0-9]+', '', url)
  url = re.sub('[_\-/]+', ' ', url)

  return ViUtils.remove_accents(url.lower()).decode()

train, test = train_test_split(df, random_state=33, test_size=0.1)
train, val = train_test_split(train, random_state=44, test_size=0.2)
cnn_train = train.copy()
cnn_x_train=cnn_train['URL']

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(cnn_x_train)

vocab_size = len(tokenizer.word_index) + 1
max_length=191
def process_texts(texts):
    sequences = tokenizer.texts_to_sequences(texts=texts)
    return pad_sequences(sequences, maxlen=max_length, padding='post')

label_encoder = LabelEncoder()
label_encoder.fit(categories)

def main():
    # st.title('URL Classification')
    input_url = st.text_input('URL')
    new_input_processed = url_to_text(input_url)

    if st.button('Categorize'):
        if input_url:
            rf_predictions = label_encoder.transform(rf_model.predict([new_input_processed]))
            xgboost_predictions = xgb_model.predict([new_input_processed])
            svm_predictions = label_encoder.transform(svm_model.predict([new_input_processed]))
            dt_predictions = label_encoder.transform(dt_model.predict([new_input_processed]))
            # nb_predictions = label_encoder.transform(nb_model.predict(x_test))
            logistic_predictions = label_encoder.transform(logis_model.predict([new_input_processed]))
            cnn_predictions = cnn_model.predict(process_texts([new_input_processed]))
            cnn_predictions = (cnn_predictions > 0.5).astype(int)
            predicted_labels = np.argmax(cnn_predictions, axis=1)

            predictions = np.column_stack((rf_predictions, xgboost_predictions, svm_predictions, dt_predictions,
                                           logistic_predictions, predicted_labels))

            # predictions = predictions.astype(int)

            final_predictions_mode, counts = mode(predictions, axis=1)
            final_predictions = final_predictions_mode.flatten()

            final_predictions_decoded = label_encoder.inverse_transform(final_predictions)

            # st.success(final_predictions_decoded)

            category = final_predictions_decoded[0]

            website_content = fetch_website_content(input_url)
            if website_content:
                # Inject ads into website content
                modified_content = inject_ads(website_content)
                # Display modified content
                st.components.v1.html(modified_content, height=800)
            else:
                st.error('Failed to fetch website content.')

            if category in gif_paths and gif_paths[category]:
                random.shuffle(gif_paths[category])
                gif_path = gif_paths[category][0]

                # Open the image to get its original dimensions
                image = Image.open(gif_path)
                original_width, original_height = image.size

                # Calculate the new width based on the desired height
                new_height = 200
                new_width = original_width

                # Resize the image while maintaining aspect ratio
                resized_image = image.resize((new_width, new_height))

                # Display the resized image
                st.image(resized_image, caption=category, width=new_width)
            else:
                st.warning("No GIF available for this category")
        else:
            st.warning("Please enter a URL")

if __name__ == '__main__':
    main()
