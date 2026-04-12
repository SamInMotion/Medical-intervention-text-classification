"""Vectorization: convert token lists into numerical feature matrices."""

from typing import List, Tuple

import numpy as np
from tensorflow.keras.preprocessing import text as keras_text


def build_vectorizer(train_texts: List[List[str]], num_words: int) -> keras_text.Tokenizer:
    """Fit a Keras tokenizer on training data."""
    tokenizer = keras_text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def vectorize(texts, tokenizer, mode="binary"):
    return tokenizer.texts_to_matrix(texts, mode=mode)


def prepare_features(train_texts, dev_texts, test_texts, num_words, mode="binary"):
    """Build vectorizer on training data and transform all splits."""
    tokenizer = build_vectorizer(train_texts, num_words)
    x_train = vectorize(train_texts, tokenizer, mode)
    x_dev = vectorize(dev_texts, tokenizer, mode)
    x_test = vectorize(test_texts, tokenizer, mode)
    return x_train, x_dev, x_test, tokenizer
