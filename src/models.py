"""Model definitions: logistic regression with and without L2.

Both models are single Dense layers with sigmoid. The thesis finding
was that this simple architecture + good features (ontology enrichment)
reaches 90%, so model complexity wasn't the bottleneck.
"""

from tensorflow.keras import Sequential, optimizers, regularizers
from tensorflow.keras.layers import Dense


def build_logistic_model(input_size):
    model = Sequential()
    model.add(Dense(units=1, activation="sigmoid", input_shape=(input_size,)))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def build_regularized_model(input_size, l2_lambda=0.000002):
    model = Sequential()
    reg = regularizers.l2(l2_lambda)
    model.add(
        Dense(
            units=1,
            activation="sigmoid",
            input_shape=(input_size,),
            kernel_regularizer=reg,
        )
    )
    opt = optimizers.Adam()
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model
