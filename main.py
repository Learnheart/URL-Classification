import pickle

import joblib
import streamlit as st
import re
from tensorflow import keras
from keras.src.utils import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from pyvi import ViUtils
import nltk
nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from collections import Counter
from keras_preprocessing.text import Tokenizer
from keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2

# stem word for svm
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english', ngram_range=(3, 3))

categories = ['Automotive', 'Books & Literature', 'Business & Finance',
       'Careers', 'Education', 'Entertainment & Art',
       'Family & Relationships', 'Food & Drink', 'Healthy Living',
       'Home & Garden', 'News & Politics', 'Real Estate', 'Science & Technology',
       'Sports', 'Style & Fashion', 'Travel', 'Games', 'Laws & Policies', 'Environment' ]

xgb_model = joblib.load('xgboost_1.4.1.pkl')
rf_model = joblib.load('random_forest_1.4.1.pkl')
dt_model = joblib.load('decision_tree_1.4.1.pkl')
nb_model = joblib.load('naive_bayes_1.4.1.pkl')
cnn_model = load_model('cnn_model_16_191.h5')

tokenizer = Tokenizer(oov_token='<OOV>')
def url_to_text(url):
  # remove stopwords
  url = url.replace('.html','').replace('.htm','').replace('http://','').replace('https://','')
  url = re.sub('^(.*?/)','/', url) # remove domains
  url = re.sub('[0-9]+', '', url)
  url = re.sub('[_\-/]+', ' ', url)

  return ViUtils.remove_accents(url.lower()).decode()

def main():
    st.title('URL Classification')
    input_url = st.text_input('URL')
    input_processed = url_to_text(input_url)
    tokenizer.fit_on_texts(input_processed)

    max_length = 191

    def process_texts(texts):
        sequences = tokenizer.texts_to_sequences(texts=texts)
        return pad_sequences(sequences, maxlen=max_length, padding='post')


    if st.button('Categorize'):
        if input_url:
            xgb_predict = xgb_model.predict([input_processed])
            rf_predict = rf_model.predict([input_processed])
            dt_predict = dt_model.predict([input_processed])
            nb_predict = nb_model.predict([input_processed])
            cnn_prediction = cnn_model.predict(process_texts([input_processed]))

            cnn_predicted_index = np.argmax(cnn_prediction, axis=1)[0]
            cnn_predicted_category = categories[cnn_predicted_index]

            all_prediction = [rf_predict[0], xgb_predict[0], dt_predict[0], nb_predict[0], cnn_predicted_category]
            final_prediction = Counter(all_prediction).most_common(1)[0][0]
            st.success(final_prediction)
        else:
            st.warning("Please enter a URL")

if __name__ == '__main__':
    main()