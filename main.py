import tensorflow
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import nltk

import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_dataset = pd.read_csv("dataset/train.csv")
y_data = train_dataset.loc[:,"sentiment"].to_numpy()

TAG_RE = re.compile(r'<[^>]+>')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = TAG_RE.sub('', text)
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, 'v') for token in text]
    text = [word for word in text if not word in stop_words]
    text = ' '.join(text)
    return text


train_dataset['Processed_Reviews'] = train_dataset.review.apply(lambda x: clean_text(x))
X_data_processed = train_dataset.loc[:,'Processed_Reviews']


print(X_data_processed.apply(lambda x: len(x.split(' '))).mean())



max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_data_processed)
X_numeric_train = tokenizer.texts_to_sequences(X_data_processed)


vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


X_numeric_train = pad_sequences(X_numeric_train, padding="post", maxlen=130)

X_train, X_val, y_train, y_val = train_test_split(X_numeric_train, y_data, test_size=0.33, random_state=42 )


model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size,128))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dropout(0.6))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fitModel = model.fit(X_train, y_train, validation_split=0.1, epochs=6, batch_size=512, verbose=1)

results = model.evaluate(X_val, y_val)
prediction = model.predict(X_val)
prediction_rounded =  np.around(prediction)
f1 = f1_score(y_val,prediction_rounded)
print(f1)


