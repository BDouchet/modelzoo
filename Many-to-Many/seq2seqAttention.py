from tensorflow.keras import models
from tensorflow.keras.layers import *

def seq2seqAttention(units,drop,
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
    encoder = LSTM(units, dropout=drop, return_state=True,return_sequences=True)
    encoder_h, state_h, state_c = encoder(input)
    if bn:
       state_h=BatchNormalization()(state_h)
       state_c=BatchNormalization()(state_c)

    #decoder
    decoder=RepeatVector(future_ts)(state_h)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=False)
    decoder_h = decoder_lstm(decoder, initial_state=[state_h,state_c])

    #context vector
    attention = dot([decoder_h, encoder_h],axes=[2,2])
    attention = Activation('softmax')(attention)
    context = dot([attention, encoder_h], axes=[2,1])
    if bn :
        context=BatchNormalization()(context)

    decoder_combined_context = concatenate([context, decoder_h])

    output = TimeDistributed(Dense(future_features))(decoder_combined_context)
    return  models.Model(input, output)
