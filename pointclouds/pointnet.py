import tensorflow as tf
from tensorflow.keras import layers,models

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


def pointnet_cls(NUM_POINTS,NUM_CLASSES):
    """
    pointnet for classification tasks
    
    NUM_POINTS := Number of points in the point cloud
    NUM_CLASSES := Number of classes to predict
    
    more info about this architecture : https://arxiv.org/abs/1612.00593
    """
    
    input=layers.Input(shape=(NUM_POINTS,3))
    
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
    
    # Global Feature
    x = layers.GlobalMaxPool1D()(x)
    
    # Classifier
    x = dense_bn(x, 512, 'relu')
    x = layers.Dropout(0.7)(x)
    x = dense_bn(x, 256, 'relu')
    x = layers.Dropout(0.7)(x)
    x = layers.Dense(NUM_CLASSES,activation='softmax')(x)
    
    return models.Model(input,x)
  
  
  
def pointnet_seg(NUM_POINTS,NUM_CLASSES):
    """
    pointnet for semantic segmentation tasks 
    
    NUM_POINTS := Number of points in the point cloud
    NUM_CLASSES := Number of classes to predict, pixel-wise
    
    more info about this architecture : https://arxiv.org/abs/1612.00593
    """
    
    input=layers.Input(shape=(NUM_POINTS,3))
    
    # input transform
    x=Tnet(input)
    x=tf.matmul(input,x)
    
    # shared mlp
    x=conv_bn(x,64,1,'relu')
    temp=conv_bn(x,64,1,'relu')
    
    # feature transform
    x=Tnet(temp)
    x=tf.matmul(temp,x)
    temp=x

    #shared mlp 
    for filter in [64,128,1024]:
        x = conv_bn(x, filter, 1,'relu')
    
    # Global Feature
    x = layers.GlobalMaxPool1D()(x)
    
    # Decoder
    x = layers.RepeatVector(NUM_POINTS)(x)
    x = layers.Concatenate()([temp,x])
    for filter in [512,256,128,128]:
        x = conv_bn(x, filter, 1,'relu')

    x = layers.Conv1D(NUM_CLASSES,1,activation='softmax')(x)
    
    return models.Model(input,x)
  
  
  
def pointnet_part(NUM_POINTS,NUM_CLASSES, NUM_PARTS):
    """
    NUM_POINTS := Number of points in the point cloud
    NUM_CLASSES := Number of classes (categories) of point cloud
    NUM_PARTS := Number of parts to perform segmentation
    
    one_hot_input take the predicted category of the point cloud from the classifier (one hot encoded)

    more info about this architecture : https://arxiv.org/abs/1612.00593
    """
    
    input=layers.Input(shape=(NUM_POINTS,3))
    
    # input transform
    x=Tnet(input)
    x=tf.matmul(input,x)
    
    # shared mlp
    temp0=conv_bn(x,64,1,'relu')
    temp1=conv_bn(temp0,128,1,'relu')
    temp2=conv_bn(temp1,128,1,'relu')
    
    # feature transform
    x=Tnet(temp2)
    temp3=tf.matmul(temp2,x)

    #shared mlp 
    temp4 = conv_bn(temp3, 512, 1,'relu')
    x = conv_bn(temp4, 2048, 1,'relu')
    
    # Global Feature
    x = layers.GlobalMaxPool1D()(x) # temp5
    
    # PartSeg
    one_hot_input = layers.Input(shape=(NUM_CLASSES,))
    one_hot_input_repeated = layers.RepeatVector(NUM_POINTS)(one_hot_input)
    x = layers.RepeatVector(NUM_POINTS)(x)
    x = layers.Concatenate()([temp0,temp1,temp2,temp3,temp4,x,one_hot_input_repeated])
    for filter in [256,256,128]:
        x = conv_bn(x, filter, 1,'relu')

    x = layers.Conv1D(NUM_PARTS,1,activation='softmax')(x)
    
    return models.Model([input,one_hot_input],x)
