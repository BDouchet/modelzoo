from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM, Reshape, Input

def VanillaLSTM(units,drop,
               back_ts,back_features,
               future_ts,future_features,
               activation='tanh'):
    """
        - units : Number of cells in the RNN
        - drop : dropout rate in the LSTM layer
        - back_ts / back_feature  : previous number of time steps / and number of features per time step (input)
        - future_ts / future_feature  : future number of time steps / and number of features per time step (output)
        - activation : activation function for the last layer
    """

    input=Input(shape=(back_ts,back_features))

    result=LSTM(units, dropout=drop, return_sequences=False)(input)

    result=Dense(future_ts*future_features, activation=activation)(result)

    result=Reshape(target_shape=(future_ts,future_features))(result)

    return models.Model(inputs=input,outputs=result)
