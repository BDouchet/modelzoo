# modelzoo

## Time Series

### Many2Many

- __vanillaLSTM__ : Encoder LSTM decoded by a Dense Layer. Single Shot RNN that predict the output sequence once.
- __seq2seq__ : A classical sequence2sequence RNN. Encoder LSTM and Decoder LSTM, so it keeps Time dependancy in the decoder/
- __seq2seqAttention__ : Similar to `seq2seq` with the add of [Luong Attention](https://arxiv.org/abs/1508.04025) 
