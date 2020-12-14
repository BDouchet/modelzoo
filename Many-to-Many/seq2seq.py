from tensorflow.keras import models
from tensorflow.keras.layers import *

def seq2seq(units,drop,
            back_ts,back_features,
            future_ts,future_features,
            bn=True, activation='tanh'):
    """
        - units : Number of cells in the RNN
        - drop : dropout rate in the LSTM layer
        - back_ts / back_feature  : previous number of time steps / and number of features per time step (input)
        - future_ts / future_feature  : future number of time steps / and number of features per time step (output)
        - bn : BatchNormalization over the encoded time series 
        - activation : activation function for the last layer
    """
    
    #input
    input = Input(shape=(back_ts,back_features))

    #encoder
    encoder = LSTM(units, dropout=drop,return_state=True)
    _, state_h, state_c = encoder(input)
    if bn:
       state_h=BatchNormalization()(state_h)
       state_c=BatchNormalization()(state_c)
    
    #decoder
    decoder=RepeatVector(future_ts)(state_h)
    decoder_lstm = LSTM(units, dropout=drop, return_sequences=True, return_state=False)
    decoder = decoder_lstm(decoder, initial_state=[state_h, state_c])
    
    #output
    output = TimeDistributed(Dense(future_features,activation=activation))(decoder)
    
    return models.Model(input, output)
