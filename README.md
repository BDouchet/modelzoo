# modelzoo

## Time Series

### Many-to-Many

- __vanillaLSTM__ : Encoder LSTM decoded by a Dense Layer. Single Shot RNN that predict the output sequence once.
- __seq2seq__ : A classical sequence2sequence RNN. Encoder LSTM and Decoder LSTM, so it keeps Time dependancy in the decoder.
- __seq2seqAttention__ : Similar to `seq2seq` with the add of [Luong Attention](https://arxiv.org/abs/1508.04025).


## 3Ds

### PointClouds
- __pointnet__ : [Network](https://arxiv.org/pdf/1612.00593.pdf) used to perform classification, semantic-segmentation and part-segmentation tasks with Points Clouds
- __pointnet_foldingnet__ : Autoencoder for points clouds data. Encoder is based on pointnet and decoder on [foldingnet](https://arxiv.org/abs/1712.07262).
- __foldingnet__ : [Foldingnet](https://arxiv.org/abs/1712.07262), autoencoder for point clouds : (B,N,3) -> (B,M,3), with M a square number


## Other

### Custom Losses

- __wcce__ : Weighted Categorical Cross Entropy, used for imbalanced classification datasets. Take a weights list in parameters.
- __dice_loss__ : Dice Loss function, mainly used in semantic segmentation
- __chamfer_loss__ : Distance used to compare a B point clouds of shape [B,N,3] with B point clouds of shape [B,M,3]
