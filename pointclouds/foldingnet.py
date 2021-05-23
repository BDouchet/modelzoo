import tensorflow as tf
from tensorflow.keras import layers,models
import tensorflow.keras.backend as K
import numpy as np
import itertools


def conv_bn(x, filter,kernel, activation):
    return layers.BatchNormalization()(layers.Conv1D(filter,kernel,activation=activation)(x))

def dense_bn(x, filter, activation):
    return layers.BatchNormalization()(layers.Dense(filter,activation=activation)(x))

def Tnet(x):
    """
    Transform Network : The first transformation network is a mini-PointNet that takes raw point cloud as input and regresses to a 3 × 3
                        matrix. It’s composed of a shared MLP(64, 128, 1024) network (with layer output sizes 64, 128, 1024) on each
                        point, a max pooling across points and two fully connected layers with output sizes 512, 256. The output matrix is
                        initialized as an identity matrix. All layers, except the last one, include ReLU and batch normalization.
    """
    out=x.shape[-1]
    for filter in [64,128,1024]:
        x = conv_bn(x, filter, 1,'relu')
    x = layers.GlobalMaxPool1D()(x)
    x = dense_bn(x, 512,'relu')
    x = dense_bn(x, 256,'relu')
    x = layers.Dense(out**2, weights=[np.zeros([256, out*out]), np.eye(out).flatten().astype(np.float32)])(x)
    return layers.Reshape((out, out))(x)

def get_grid(m):
    """
    return the M coordinates of a m x m square regular grid between -1 and 1 (m²=M). 
    output shape : [1,M,2] 
    """
    lin = np.linspace(-1,1,m)
    grid = np.expand_dims(np.array(list(itertools.product(lin, lin))),axis=0)
    return grid

def foldingnet(N, M, grid):
    """
    FoldingNet Decoder with Pointnet Encoder
    
    N := Number of points in the input point cloud
    M := Number of points in the output point cloud
    grid := (M, 2) array of coordinates of the grid.

    more info about this architecture : https://arxiv.org/abs/1712.07262
    """
    input=layers.Input(shape=(N,3))
    
    ### ENCODER POINTNET ###
    # input transform
    x=Tnet(input)
    x=tf.matmul(input,x)
    
    # shared mlp
    x=conv_bn(x,64,1,'relu')
    temp=conv_bn(x,64,1,'relu')
    
    # feature transform
    x=Tnet(temp)
    x=tf.matmul(temp,x)
    
    #shared mlp 
    for filter in [64,128,1024]:
        x = conv_bn(x, filter, 1,'relu')
    
    ### GLOBAL FEATURE ###
    x = layers.GlobalMaxPool1D()(x)
    x = dense_bn(x, 1024,'relu')
    x = dense_bn(x, 512,'relu')

    
    ### DECODER FOLDINGNET ###
    glob = layers.RepeatVector(M)(x)
    grid = K.tile(grid,n=(K.shape(glob)[0],1,1))
    x = layers.Concatenate()([glob,grid])  

    x = conv_bn(x, 514, 1, activation='relu')
    x = conv_bn(x, 512, 1, activation='relu')
    x = conv_bn(x, 3, 1, activation='linear')
    
    x = layers.Concatenate()([glob,x])
    x = conv_bn(x, 517, 1, activation='relu')
    x = conv_bn(x, 512, 1, activation='relu')
    x = conv_bn(x, 3, 1, activation='linear')
       
    return models.Model(input,x)
