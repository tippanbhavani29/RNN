import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 500

model = tf.keras.models.load_model(
    "../models/sentiment_rnn.keras"
)

with open(
    "../models/word_index.json",
    "r"
) as f:

    word_index = json.load(f)


def encode_review(review):

    review = review.lower()

    words = review.split()

    encoded = []

    for word in words:

        if word in word_index:

            encoded.append(
                word_index[word] + 3
            )

        else:

            encoded.append(2)

    return encoded


def predict_sentiment(review):

    encoded_review = encode_review(
        review
    )

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN
    )

    prediction = model.predict(
        padded_review,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if prediction >= 0.5
        else "Negative"
    )

    confidence = (
        prediction * 100
        if prediction >= 0.5
        else (1 - prediction) * 100
    )

    return sentiment, confidence