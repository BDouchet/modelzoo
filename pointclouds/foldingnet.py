import tensorflow as tf
from tensorflow.keras import layers,models
import tensorflow.keras.backend as K
import numpy as np
import itertools


def conv_bn(x, filter,kernel, activation):
    return layers.BatchNormalization()(layers.Conv1D(filter,kernel,activation=activation)(x))

def dense_bn(x, filter, activation):
    return layers.BatchNormalization()(layers.Dense(filter,activation=activation)(x))


def get_grid(m):
    lin = np.linspace(-1,1,m)
    grid = np.expand_dims(np.array(list(itertools.product(lin, lin))),axis=0)
    return grid

def knn(x,k):
    x=tf.convert_to_tensor(x,dtype='float32')                   # (B,N,3)
    difference = (
        tf.expand_dims(x, axis=-2) -
        tf.expand_dims(x, axis=-3))                             # (B,N,N,3)
    square_distances = - tf.einsum("...i,...i->...",            # (B,N,N)
                                   difference, difference)
    _,topk=tf.math.top_k(square_distances, k=k, sorted=True)    # (B,N,k)
    return tf.expand_dims(topk,axis=3)                          # (B,N,k,1)

def local_cov(x,idx):
    # x : (B,N,3) - idx : (B,N,k,1)
    pts=tf.gather_nd(x,idx,batch_dims=1)                        # (B,N,k,3)
    mat=pts-tf.reduce_mean(pts,axis=2,keepdims=True)            # (B,N,k,3)
    cov=tf.linalg.matmul(mat,mat,transpose_a=True)              # (B,N,3,3)
    return tf.reshape(cov,(-1,x.shape[1],9))                    # (B,N,9)

def local_maxpool(x, idx):
    # x : (B,N,feature) - idx : (B,N,k,1)
    pts=tf.gather_nd(x,idx,batch_dims=1)                        # (B,N,k,feature)
    return tf.reduce_max(pts,axis=2)                            # (B,N,feature)

def foldingnet(N, M, grid):
    """
    FoldingNet Decoder with Pointnet Encoder
    
    N := Number of points in the input point cloud
    M := Number of points in the output point cloud
    grid := (M, 2) array of coordinates of the grid.

    more info about this architecture : https://arxiv.org/abs/1712.07262
    """
    input = layers.Input(shape=(N,3))                           # (B,N,3)
    
    graph = knn(input,k=16)                                     # (B,N,k,1)
    cov = local_cov(input,graph)                                # (B,N,9)
    x = layers.Concatenate()([input,cov])                       # (B,N,12)

    ### ENCODER Foldingnet ###
    # MLP 1
    for _ in range(3):
        x = conv_bn(x, filter=64, kernel=1, activation='relu')  # (B,N,64)
    
    # Graph Layer
    x = local_maxpool(x, graph)                                 # (B,N,64)
    x = conv_bn(x, filter=64, kernel=1, activation='relu')      # (B,N,64)
    x = conv_bn(x, filter=128, kernel=1, activation='relu')     # (B,N,128)
    x = local_maxpool(x, graph)                                 # (B,N,128)
    x = conv_bn(x, filter=128, kernel=1, activation='relu')     # (B,N,128)
    x = conv_bn(x, filter=1024, kernel=1, activation='relu')    # (B,N,1024)
    
    # Global Max Pooling 
    x = layers.GlobalMaxPool1D()(x)                             # (B,1024)

    # MLP 2
    x = dense_bn(x, filter=512, activation='relu')              # (B,512)
    x = dense_bn(x, filter=512, activation='linear')            # (B,512)

    
    ### DECODER FOLDINGNET ###
    glob = layers.RepeatVector(M)(x)                            # (B,M,512)
    grid = K.tile(grid,n=(K.shape(glob)[0],1,1))                # (B,M,2)
    x = layers.Concatenate()([glob,grid])                       # (B,M,514)

    x = conv_bn(x, 512, 1, activation='relu')                   # (B,M,512)
    x = conv_bn(x, 512, 1, activation='relu')                   # (B,M,512)
    x = conv_bn(x, 3, 1, activation='linear')                   # (B,M,3)
    
    x = layers.Concatenate()([glob,x])                          # (B,M,515)
    x = conv_bn(x, 512, 1, activation='relu')                   # (B,M,512)
    x = conv_bn(x, 512, 1, activation='relu')                   # (B,M,512)
    x = conv_bn(x, 3, 1, activation='tanh')                     # (B,M,3)
       
    return models.Model(input,x)
