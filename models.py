import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import LSTM, Dense, Flatten, Input, Dropout, BatchNormalization, Concatenate, Conv1D, MaxPool1D, Dot, Layer, Embedding
from utils.layers import Conv1DTranspose, UNetModule

VOCAB_SIZE = 22


def embedding():
    embedding = tf.Variable(initial_value = np.load('../Embedding/embedd_w.npy').reshape(1,21,10), trainable= True)
    inp    = Input(shape=(1000,21))
    tct0_in =  Dot(axes=(2,1))([inp,embedding])
    model = Model(inputs = inp, outputs = tct0_in)
    
    return model
    
def u_net_module_hp(hp):
    
############################## Special Fun ###################################################

   # embedding = tf.Variable(initial_value = np.load('../Embedding/embedd_w_5_ogt.npy').reshape(1,21,5))
    
############################## Transformation module #################################################
    
    p = {'n_fil_1':hp.Choice('number_filters_conv1', [8, 16, 32, 64]),
         's_fil_1':hp.Choice('size_filters_conv1',[3, 5, 7, 10]),
         'stride_1':hp.Choice('stride_length_sampling1', [2,4]),
         'dOut_1':hp.Choice('Dropout_module1',[0.1, 0.2, 0.3, 0.4]),
         'n_fil_2':hp.Choice('number_filters_conv2', [32, 64, 128]),
         's_fil_2':hp.Choice('size_filters_conv2',[3, 5, 7, 10]),
         'stride_2':hp.Choice('stride_length_sampling2', [2,4]),
         'dOut_2':hp.Choice('Dropout_module2',[0.1, 0.2, 0.3, 0.4]),
         'n_fil_3':hp.Choice('number_filters_conv3', [64, 128, 256]),
         's_fil_3':hp.Choice('size_filters_conv3',[3, 5, 7, 10]),
         'stride_3':hp.Choice('stride_length_sampling3', [2,4]),
         'dOut_3':hp.Choice('Dropout_module3',[0.1, 0.2, 0.3, 0.4]),
         'n_fil_4':hp.Choice('number_filters_conv4', [128, 256, 512]),
         's_fil_4':hp.Choice('size_filters_conv4',[3, 5, 7, 10]),
         'dOut_4':hp.Choice('Dropout_module4',[0.1, 0.2, 0.3, 0.4]),
         'dOut_5':hp.Choice('Dropout_module5',[0.05, 0.1, 0.2, 0.3]),
         's_fil_5':hp.Choice('size_filters_conv5',[3, 5, 7, 10])
        }
    
    # Layers of stage 0 contraction
    
    inp    = Input(shape=(1024,21))
                 
    #tct0_in =  Dot(axes=(2,1))([inp,embedding])
    tct0_bn1   = BatchNormalization()(inp)#tct0_in)
    #tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
    tct0_conv1 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Ct0_0')(tct0_bn1)
    tct0_bn2   = BatchNormalization()(tct0_conv1)
    tct0_conv2 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Ct0_1')(tct0_bn2)
    tct0_bn3   = BatchNormalization()(tct0_conv2)
    tct0_conv3 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu',strides = int(p['stride_1']), padding='same', name='Convolution_Ct0_2')(tct0_bn3)
    tct0_bn4   = BatchNormalization()(tct0_conv3)
    #tct0_max   = MaxPool1D(pool_size=2, strides=2)(tct0_bn2)
    tct0_dp    = Dropout(p['dOut_1'])(tct0_bn4)
    
    # Layers of stage 1 contraction
    
    tct1_conv1 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Ct1_0')(tct0_dp)
    tct1_bn1   = BatchNormalization()(tct1_conv1)
    tct1_conv2 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', strides=1, padding='same', name='Convolution_Ct1_1')(tct1_bn1)
    tct1_bn2   = BatchNormalization()(tct1_conv2)
    tct1_conv3 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', strides=int(p['stride_2']), padding='same', name='Convolution_Ct1_2')(tct1_bn2)
    tct1_bn3   = BatchNormalization()(tct1_conv3)
    #tct1_max   = MaxPool1D(pool_size=2, strides=2)(tct1_bn2)
    tct1_dp    = Dropout(p['dOut_2'])(tct1_bn3)
    
    # Layers of stage 2 contraction
    
    tct2_conv1 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Ct2_0')(tct1_dp)
    tct2_bn1   = BatchNormalization()(tct2_conv1)
    tct2_conv2 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', strides=1, padding='same', name='Convolution_Ct2_1')(tct2_bn1)
    tct2_bn2   = BatchNormalization()(tct2_conv2)
    tct2_conv3 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', strides=int(p['stride_3']), padding='same', name='Convolution_Ct2_2')(tct2_bn2)
    tct2_bn3   = BatchNormalization()(tct2_conv3)
    #tct2_max   = MaxPool1D(pool_size=2, strides=2)(tct2_bn2)
    tct2_dp    = Dropout(p['dOut_3'])(tct2_bn3)
    
    # Layers of stage 3 contraction
    
    tct3_conv1 = Conv1D(int(p['n_fil_4']), int(p['s_fil_4']), activation='relu', padding='same', name='Convolution_Ce3_0')(tct2_dp)
    tct3_bn1   = BatchNormalization()(tct3_conv1)
    tct3_conv2 = Conv1D(int(p['n_fil_4']), int(p['s_fil_4']), activation='relu', padding='same', name='Convolution_Ce3_1')(tct3_bn1)
    tct3_bn2   = BatchNormalization()(tct3_conv2)
    tct3_dp    = Dropout(p['dOut_4'])(tct3_bn2)
    
    # Layers of stage 1 expansion
    
    tet1_Tconv  = Conv1DTranspose(int(p['n_fil_3']), int(p['s_fil_3']), strides=int(p['stride_3']) ,activation='relu', padding='same', name='TransConv_Et1')(tct3_dp)
    tet1_Concat = Concatenate(axis=2)([tet1_Tconv, tct2_conv1])
    tet1_bn1    = BatchNormalization()(tet1_Concat)
    tet1_conv1  = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Et1_0')(tet1_bn1)
    tet1_bn2    = BatchNormalization()(tet1_conv1)
    tet1_conv2  = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Et1_1')(tet1_bn2)
    tet1_bn3    = BatchNormalization()(tet1_conv2)
    tet1_dp     = Dropout(p['dOut_3'])(tet1_bn3)
    
    #Layers of stage 2 expansion
               
    tet2_Tconv  = Conv1DTranspose(int(p['n_fil_2']), int(p['s_fil_2']), strides=int(p['stride_2']) ,activation='relu', padding='same', name='TransConv_Et2')(tet1_dp)
    tet2_Concat = Concatenate(axis=2)([tet2_Tconv, tct1_conv1])
    tet2_bn1    = BatchNormalization()(tet2_Concat)
    tet2_conv1  = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Et2_0')(tet2_bn1)
    tet2_bn2    = BatchNormalization()(tet2_conv1)
    tet2_conv2  = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Et2_1')(tet2_bn2)
    tet2_bn3    = BatchNormalization()(tet2_conv2)
    tet2_dp     = Dropout(p['dOut_2'])(tet2_bn3)
                       
    #Layers of stage 3 expansion
               
    tet3_Tconv = Conv1DTranspose(int(p['n_fil_1']), int(p['s_fil_1']), strides=int(p['stride_1']) ,activation='relu', padding='same', name='TransConv_Et3')(tet2_dp)
    tet3_Concat = Concatenate(axis=2)([tet3_Tconv, tct0_conv1])
    tet3_bn1 = BatchNormalization()(tet3_Concat)
    tet3_conv1 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Et3_1')(tet3_bn1)
    tet3_bn2 = BatchNormalization()(tet3_conv1)
    tet3_conv2 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Et3_2')(tet3_bn2)
    tet3_bn3 = BatchNormalization()(tet3_conv2)
    tet3_dp = Dropout(p['dOut_5'])(tet3_bn3)
    tet3_conv3 = Conv1D(3, int(p['s_fil_5']), activation='softmax', padding='same', name='Convolution_Et3_3')(tet3_dp)

    model = Model(inputs = inp, outputs = tet3_conv3)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True) #, sample_weight = )
    add = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]),name='Adam')
    model.compile(optimizer = add, loss = cce, metrics=['accuracy']) 
    
    return model

def u_net(p):
    
    reg = L1L2(l1 = p['l1'][-1], l2 = p['l2'][-1])   
    inp = Input(shape=(p['max_seq_len'],21))
    x = Embedding(VOCAB_SIZE, p['emb_size'])(inp)
    x = UNetModule(p)(inp)
    out = Conv1D(p['num_classes'], p['kernel_size'][-1], activation='softmax', padding='same',
                 kernel_regularizer = reg, bias_regularizer = reg)(x)

    model = Model(inputs = inp, outputs = out)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)# , sample_weight = w)
    add = tf.keras.optimizers.Adam(learning_rate=p['learning_rate'],name='Adam')
    model.compile(optimizer = add, loss = cce, metrics=['accuracy'], sample_weight_mode="temporal") 
    
    return model

def u_net_module(p):
    
############################## Special Fun ###################################################

   # embedding = tf.Variable(initial_value = np.load('../Embedding/embedd_w_5_ogt.npy').reshape(1,21,5))
    
############################## Transformation module #################################################
    

    
    # Layers of stage 0 contraction
    
    inp    = Input(shape=(1024,))
    #w = Input(shape=(1024,))
                 
    #tct0_in =  Dot(axes=(2,1))([inp,embedding])
    #tct0_bn1   = BatchNormalization()(inp)#tct0_in)
    #tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)
    x = Embedding(22, 5)(inp)
    x = BatchNormalization()(x)
    tct0_conv1 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Ct0_0')(x)
    tct0_bn2   = BatchNormalization()(tct0_conv1)
    tct0_conv2 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Ct0_1')(tct0_bn2)
    tct0_bn3   = BatchNormalization()(tct0_conv2)
    tct0_conv3 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu',strides = int(p['stride_1']), padding='same', name='Convolution_Ct0_2')(tct0_bn3)
    tct0_bn4   = BatchNormalization()(tct0_conv3)
    #tct0_max   = MaxPool1D(pool_size=2, strides=2)(tct0_bn2)
    tct0_dp    = Dropout(p['dOut_1'])(tct0_bn4)
    
    # Layers of stage 1 contraction
    
    tct1_conv1 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Ct1_0')(tct0_dp)
    tct1_bn1   = BatchNormalization()(tct1_conv1)
    tct1_conv2 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', strides=1, padding='same', name='Convolution_Ct1_1')(tct1_bn1)
    tct1_bn2   = BatchNormalization()(tct1_conv2)
    tct1_conv3 = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', strides=int(p['stride_2']), padding='same', name='Convolution_Ct1_2')(tct1_bn2)
    tct1_bn3   = BatchNormalization()(tct1_conv3)
    #tct1_max   = MaxPool1D(pool_size=2, strides=2)(tct1_bn2)
    tct1_dp    = Dropout(p['dOut_2'])(tct1_bn3)
    
    # Layers of stage 2 contraction
    
    tct2_conv1 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Ct2_0')(tct1_dp)
    tct2_bn1   = BatchNormalization()(tct2_conv1)
    tct2_conv2 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', strides=1, padding='same', name='Convolution_Ct2_1')(tct2_bn1)
    tct2_bn2   = BatchNormalization()(tct2_conv2)
    tct2_conv3 = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', strides=int(p['stride_3']), padding='same', name='Convolution_Ct2_2')(tct2_bn2)
    tct2_bn3   = BatchNormalization()(tct2_conv3)
    #tct2_max   = MaxPool1D(pool_size=2, strides=2)(tct2_bn2)
    tct2_dp    = Dropout(p['dOut_3'])(tct2_bn3)
    
    # Layers of stage 3 contraction
    
    tct3_conv1 = Conv1D(int(p['n_fil_4']), int(p['s_fil_4']), activation='relu', padding='same', name='Convolution_Ce3_0')(tct2_dp)
    tct3_bn1   = BatchNormalization()(tct3_conv1)
    tct3_conv2 = Conv1D(int(p['n_fil_4']), int(p['s_fil_4']), activation='relu', padding='same', name='Convolution_Ce3_1')(tct3_bn1)
    tct3_bn2   = BatchNormalization()(tct3_conv2)
    tct3_dp    = Dropout(p['dOut_4'])(tct3_bn2)
    
    # Layers of stage 1 expansion
    
    tet1_Tconv  = Conv1DTranspose(int(p['n_fil_3']), int(p['s_fil_3']), strides=int(p['stride_3']), name='TransConv_Et1')(tct3_dp)
    tet1_Concat = Concatenate(axis=2)([tet1_Tconv, tct2_conv1])
    tet1_bn1    = BatchNormalization()(tet1_Concat)
    tet1_conv1  = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Et1_0')(tet1_bn1)
    tet1_bn2    = BatchNormalization()(tet1_conv1)
    tet1_conv2  = Conv1D(int(p['n_fil_3']), int(p['s_fil_3']), activation='relu', padding='same', name='Convolution_Et1_1')(tet1_bn2)
    tet1_bn3    = BatchNormalization()(tet1_conv2)
    tet1_dp     = Dropout(p['dOut_3'])(tet1_bn3)
    
    #Layers of stage 2 expansion
               
    tet2_Tconv  = Conv1DTranspose(int(p['n_fil_2']), int(p['s_fil_2']), strides=int(p['stride_2']) , name='TransConv_Et2')(tet1_dp)
    tet2_Concat = Concatenate(axis=2)([tet2_Tconv, tct1_conv1])
    tet2_bn1    = BatchNormalization()(tet2_Concat)
    tet2_conv1  = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Et2_0')(tet2_bn1)
    tet2_bn2    = BatchNormalization()(tet2_conv1)
    tet2_conv2  = Conv1D(int(p['n_fil_2']), int(p['s_fil_2']), activation='relu', padding='same', name='Convolution_Et2_1')(tet2_bn2)
    tet2_bn3    = BatchNormalization()(tet2_conv2)
    tet2_dp     = Dropout(p['dOut_2'])(tet2_bn3)
                       
    #Layers of stage 3 expansion
               
    tet3_Tconv = Conv1DTranspose(int(p['n_fil_1']), int(p['s_fil_1']), strides=int(p['stride_1']) , name='TransConv_Et3')(tet2_dp)
    tet3_Concat = Concatenate(axis=2)([tet3_Tconv, tct0_conv1])
    tet3_bn1 = BatchNormalization()(tet3_Concat)
    tet3_conv1 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Et3_1')(tet3_bn1)
    tet3_bn2 = BatchNormalization()(tet3_conv1)
    tet3_conv2 = Conv1D(int(p['n_fil_1']), int(p['s_fil_1']), activation='relu', padding='same', name='Convolution_Et3_2')(tet3_bn2)
    tet3_bn3 = BatchNormalization()(tet3_conv2)
    tet3_dp = Dropout(p['dOut_5'])(tet3_bn3)
    tet3_conv3 = Conv1D(3, int(p['s_fil_5']), activation='softmax', padding='same', name='Convolution_Et3_3')(tet3_dp)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)# , sample_weight = w)
    add = tf.keras.optimizers.Adam(learning_rate=1e-2,name='Adam')
    model = Model(inputs = inp, outputs = tet3_conv3)
    model.compile(optimizer = add, loss = cce, metrics=['accuracy'], sample_weight_mode="temporal") 
    
    return model

def u_net_module_2(in_):
    
############################## Special Fun ###################################################

    embedding = tf.Variable(initial_value = np.load('../Embedding/embedd_w_5.npy').reshape(1,21,5), trainable= True)
    
############################## Transformation module #################################################
    
    # Layers of stage 0 contraction
    
    inp    = InputLayer(input_shape=(in_,21))
                 
    tct0_in =  Dot(axes=(2,1))([inp,embedding])
    tct0_bn1   = BatchNormalization()(tct0_in)
    tct0_conv1 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct0_0')(tct0_bn1)
    tct0_bn2   = BatchNormalization()(tct0_conv1)
    tct0_conv2 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct0_1')(tct0_bn2)
    tct0_bn3   = BatchNormalization()(tct0_conv2)
    tct0_conv3 = Conv1D(32, 5, activation='relu',strides = 2, padding='same', name='Convolution_Ct0_2')(tct0_bn3)
    tct0_bn4   = BatchNormalization()(tct0_conv3)
    #tct0_max   = MaxPool1D(pool_size=2, strides=2)(tct0_bn2)
    tct0_dp    = Dropout(0.2)(tct0_bn4)
    
    # Layers of stage 1 contraction
    
    tct1_conv1 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct1_0')(tct0_dp)
    tct1_bn1   = BatchNormalization()(tct1_conv1)
    tct1_conv2 = Conv1D(32, 5, activation='relu', strides=1, padding='same', name='Convolution_Ct1_1')(tct1_bn1)
    tct1_bn2   = BatchNormalization()(tct1_conv2)
    tct1_conv3 = Conv1D(32, 5, activation='relu', strides=2, padding='same', name='Convolution_Ct1_2')(tct1_bn2)
    tct1_bn3   = BatchNormalization()(tct1_conv3)
    #tct1_max   = MaxPool1D(pool_size=2, strides=2)(tct1_bn2)
    tct1_dp    = Dropout(0.2)(tct1_bn3)
    
    # Layers of stage 2 contraction
    
    tct2_conv1 = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Ct2_0')(tct1_dp)
    tct2_bn1   = BatchNormalization()(tct2_conv1)
    tct2_conv2 = Conv1D(64, 5, activation='relu', strides=1, padding='same', name='Convolution_Ct2_1')(tct2_bn1)
    tct2_bn2   = BatchNormalization()(tct2_conv2)
    tct2_conv3 = Conv1D(64, 5, activation='relu', strides=2, padding='same', name='Convolution_Ct2_2')(tct2_bn2)
    tct2_bn3   = BatchNormalization()(tct2_conv3)
    #tct2_max   = MaxPool1D(pool_size=2, strides=2)(tct2_bn2)
    tct2_dp    = Dropout(0.2)(tct2_bn3)
    
    # Layers of stage 3 contraction
    
    tct3_conv1 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_0')(tct2_dp)
    tct3_bn1   = BatchNormalization()(tct3_conv1)
    tct3_conv2 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_1')(tct3_bn1)
    tct3_bn2   = BatchNormalization()(tct3_conv2)
    tct3_dp    = Dropout(0.2)(tct3_bn2)
    
    # Layers of stage 1 expansion
    
    tet1_Tconv  = Conv1DTranspose(64, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et1')(tct3_dp)
    tet1_Concat = Concatenate(axis=2)([tet1_Tconv, tct2_conv1])
    tet1_bn1    = BatchNormalization()(tet1_Concat)
    tet1_conv1  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_0')(tet1_bn1)
    tet1_bn2    = BatchNormalization()(tet1_conv1)
    tet1_conv2  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_1')(tet1_bn2)
    tet1_bn3    = BatchNormalization()(tet1_conv2)
    tet1_dp     = Dropout(0.2)(tet1_bn3)
    
    #Layers of stage 2 expansion
               
    tet2_Tconv  = Conv1DTranspose(32, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et2')(tet1_dp)
    tet2_Concat = Concatenate(axis=2)([tet2_Tconv, tct1_conv1])
    tet2_bn1    = BatchNormalization()(tet2_Concat)
    tet2_conv1  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_0')(tet2_bn1)
    tet2_bn2    = BatchNormalization()(tet2_conv1)
    tet2_conv2  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_1')(tet2_bn2)
    tet2_bn3    = BatchNormalization()(tet2_conv2)
    tet2_dp     = Dropout(0.2)(tet2_bn3)
                       
    #Layers of stage 3 expansion
               
    tet3_Tconv = Conv1DTranspose(32, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et3')(tet2_dp)
    tet3_Concat = Concatenate(axis=2)([tet3_Tconv, tct0_conv1])
    tet3_bn1 = BatchNormalization()(tet3_Concat)
    tet3_conv1 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et3_1')(tet3_bn1)
    tet3_bn2 = BatchNormalization()(tet3_conv1)
    tet3_conv2 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et3_2')(tet3_bn2)
    tet3_bn3 = BatchNormalization()(tet3_conv2)
    tet3_dp = Dropout(0.1)(tet3_bn3)
    tet3_conv3 = Conv1D(3, 5, activation='softmax', padding='same', name='Convolution_Et3_3')(tet3_dp)
    
    ################################################ Stage 2 ######################################################
    
    tet3_Concat2 = Concatenate(axis=2)([tct0_in, tet3_conv3])
    tct0_bn12   = BatchNormalization()(tet3_Concat2)
    tct0_conv12 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct0_02')(tct0_bn12)
    tct0_bn22  = BatchNormalization()(tct0_conv12)
    tct0_conv22 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct0_12')(tct0_bn22)
    tct0_bn32   = BatchNormalization()(tct0_conv22)
    tct0_conv32 = Conv1D(32, 5, activation='relu',strides = 2, padding='same', name='Convolution_Ct0_22')(tct0_bn32)
    tct0_bn42   = BatchNormalization()(tct0_conv32)
    #tct0_max   = MaxPool1D(pool_size=2, strides=2)(tct0_bn2)
    tct0_dp2    = Dropout(0.2)(tct0_bn42)
    
    # Layers of stage 1 contraction
    
    tct1_conv12 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct1_02')(tct0_dp2)
    tct1_bn12   = BatchNormalization()(tct1_conv12)
    tct1_conv22 = Conv1D(32, 5, activation='relu', strides=1, padding='same', name='Convolution_Ct1_12')(tct1_bn12)
    tct1_bn22   = BatchNormalization()(tct1_conv22)
    tct1_conv32 = Conv1D(32, 5, activation='relu', strides=2, padding='same', name='Convolution_Ct1_22')(tct1_bn22)
    tct1_bn32   = BatchNormalization()(tct1_conv32)
    #tct1_max   = MaxPool1D(pool_size=2, strides=2)(tct1_bn2)
    tct1_dp2    = Dropout(0.2)(tct1_bn32)
    
    # Layers of stage 2 contraction
    
    tct2_conv12 = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Ct2_02')(tct1_dp2)
    tct2_bn12   = BatchNormalization()(tct2_conv12)
    tct2_conv22 = Conv1D(64, 5, activation='relu', strides=1, padding='same', name='Convolution_Ct2_12')(tct2_bn12)
    tct2_bn22   = BatchNormalization()(tct2_conv22)
    tct2_conv32 = Conv1D(64, 5, activation='relu', strides=2, padding='same', name='Convolution_Ct2_22')(tct2_bn22)
    tct2_bn32   = BatchNormalization()(tct2_conv32)
    #tct2_max   = MaxPool1D(pool_size=2, strides=2)(tct2_bn2)
    tct2_dp2    = Dropout(0.2)(tct2_bn32)
    
    # Layers of stage 3 contraction
    
    tct3_conv12 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_02')(tct2_dp2)
    tct3_bn12   = BatchNormalization()(tct3_conv12)
    tct3_conv22 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_12')(tct3_bn12)
    tct3_bn22   = BatchNormalization()(tct3_conv22)
    tct3_dp2    = Dropout(0.2)(tct3_bn22)
    
    # Layers of stage 1 expansion
    
    tet1_Tconv2  = Conv1DTranspose(64, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et12')(tct3_dp2)
    tet1_Concat2 = Concatenate(axis=2)([tet1_Tconv2, tct2_conv12])
    tet1_bn12    = BatchNormalization()(tet1_Concat2)
    tet1_conv12  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_02')(tet1_bn12)
    tet1_bn22    = BatchNormalization()(tet1_conv12)
    tet1_conv22  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_12')(tet1_bn22)
    tet1_bn32    = BatchNormalization()(tet1_conv22)
    tet1_dp2     = Dropout(0.2)(tet1_bn32)
    
    #Layers of stage 2 expansion
               
    tet2_Tconv2  = Conv1DTranspose(32, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et22')(tet1_dp2)
    tet2_Concat2 = Concatenate(axis=2)([tet2_Tconv2, tct1_conv12])
    tet2_bn12    = BatchNormalization()(tet2_Concat2)
    tet2_conv12  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_02')(tet2_bn12)
    tet2_bn22    = BatchNormalization()(tet2_conv12)
    tet2_conv22  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_12')(tet2_bn22)
    tet2_bn32    = BatchNormalization()(tet2_conv22)
    tet2_dp2     = Dropout(0.2)(tet2_bn32)
                       
    #Layers of stage 3 expansion
               
    tet3_Tconv2 = Conv1DTranspose(32, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et32')(tet2_dp2)
    tet3_Concat2 = Concatenate(axis=2)([tet3_Tconv2, tct0_conv12])
    tet3_bn12 = BatchNormalization()(tet3_Concat2)
    tet3_conv12 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et3_12')(tet3_bn12)
    tet3_bn22 = BatchNormalization()(tet3_conv12)
    tet3_conv22 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et3_22')(tet3_bn22)
    tet3_bn32 = BatchNormalization()(tet3_conv22)
    tet3_dp2 = Dropout(0.1)(tet3_bn32)
    tet3_conv32 = Conv1D(3, 5, activation='softmax', padding='same', name='Convolution_Et3_32')(tet3_dp2)

    model = Model(inputs = inp, outputs = [tet3_conv3, tet3_conv32])
    
    return model


def lstm_cyclic_gan():
    input_x = Input(shape = (None,None,20))
    input_dist_target = Input(shape = (None,None,20))
    
    encoding_layer_1 = CuDNNLSTM(256, return_sequences=True, return_state=True)
    encoding_layer_2 = CuDNNLSTM(256, return_sequences=False, return_state=False)
    encoding_layer_3 = CuDNNLSTM(256, return_sequences=False, return_state=False)
    
    decoder_layer_1 = CuDNNLSTM(256, return_sequences=True, return_state=True)
    decoder_layer_2 = CuDNNLSTM(256, return_sequences=True, return_state=True)
    
    #discriminator_lstm = CuDNNLSTM(256,return_sequences=False, return_states=False)
    discriminator_dense1 = Dense(128, activation='relu')
    discriminator_batchNorm = BatchNormalization()
    discriminator_dropout = Dropout(0.4)
    discriminator_dense2 = Dense(1, activation=None)
    
    
    x_decode_hidden, x_decode_state = encoding_layer(input_x)
    transformed_seq, transformed_state = decoder_layer_1(inputs = x_decode_hidden, initial_state=x_decode_state)
    restored_seq = decoder_layer_2(inputs = transformed_seq, initial_state=_transformed_state)
     
    
    encode_fake = encoding_layer_2(inputs = x_decode_hidden, initial_state = x_decode_state)
    encode_real = encoding_layer_3(inputs = inputs_dist_target, initial_state = None)
    concat = Concatinate()([encode])
    X = discriminator_dense1()
    
    model_encode_decode = Model(inputs = x_inputs, outputs = restored_seq)
    
    
    
def cnn_cycle_gan():
    
    ############################## Transformation module #################################################
    
    # Layers of stage 0 contraction
    
    InputLayer(input_shape=(in_,21))
    tct0_conv1 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Ct0_1')(tct0_in)
    tct0_bn1   = BatchNormalization()(ct0_conv1)
    tct0_conv2 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Ct0_2')(tct0_bn1)
    tct0_bn2   = BatchNormalization()(tct0_conv2)
    tct0_max   = MaxPool1D(pool_size=2, strides=2)(tct0_bn2)
    tct0_dp    = Dropout(0.2)(tct0_max)
    
    # Layers of stage 1 contraction
    
    tct1_conv1 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct1_1')(tct0_dp)
    tct1_bn1   = BatchNormalization()(tct1_conv1)
    tct1_conv2 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct1_2')(tct1_bn1)
    tct1_bn2   = BatchNormalization()(tct1_conv2)
    tct1_max   = MaxPool1D(pool_size=2, strides=2)(tct1_bn2)
    tct1_dp    = Dropout(0.2)(tct1_max)
    
    # Layers of stage 2 contraction
    
    tct2_conv1 = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Ct2_1')(tct1_dp)
    tct2_bn1   = BatchNormalization()(tct2_conv1)
    tct2_conv2 = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Ct2_2')(tct2_bn1)
    tct2_bn2   = BatchNormalization()(tct2_conv2)
    tct2_max   = MaxPool1D(pool_size=2, strides=2)(tct2_bn2)
    tct2_dp    = Dropout(0.2)(tct2_max)
    
    # Layers of stage 3 contraction
    
    tct3_conv1 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_1')(tct2_dp)
    tct3_bn1   = BatchNormalization()(tct3_conv1)
    tct3_conv2 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_2')(tct3_bn1)
    tct3_bn2   = BatchNormalization()(tct3_conv2)
    tct3_max   = MaxPool1D(pool_size=2, strides=2)(tct3_bn2)
    tct3_dp    = Dropout(0.2)(tct3_max)
    
    # Layers of stage 1 expansion
    
    tet1_Tconv  = Conv1DTranspose(64, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et1')(tct3_dp)
    tet1_Concat = Concatenate(axis=1)([tet1_Tconv, tct3_conv2])
    tet1_bn1    = BatchNormalization()(et1_Concat)
    tet1_conv1  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_1')(tet1_bn1)
    tet1_bn2    = BatchNormalization()(tet1_conv1)
    tet1_conv2  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_2')(tet1_bn2)
    tet1_dp     = Dropout(0.2)(tet1_conv2)
    
    #Layers of stage 2 expansion
               
    tet2_Tconv  = Conv1DTranspose(32, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et2')(tet1_dp)
    tet2_Concat = Concatenate(axis=1)([tet2_Tconv, tct1_conv2])
    tet2_bn1    = BatchNormalization()(tet2_Concat)
    tet2_conv1  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_1')(tet2_bn1)
    tet2_bn2    = BatchNormalization()(tet2_conv1)
    tet2_conv2  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_2')(tet2_bn2)
    tet2_dp     = Dropout(0.2)(tet2_conv2)
                       
    #Layers of stage 3 expansion
               
    tet3_Tconv = Conv1DTranspose(16, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et3')(tet2_dp)
    tet3_Concat = Concatenate(axis=1)([tet3_Tconv, tce0_conv2])
    tet3_bn1 = BatchNormalization()(tet3_Concat)
    tet3_conv1 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Et3_1')(tet3_bn1)
    tet3_bn2 = BatchNormalization()(tet3_conv1)
    tet3_conv2 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Et3_2')(tet3_bn2)
    tet3_bn3 = BatchNormalization()(tet3_conv2)
    tet3_dp = Dropout(0.1)(tet3_bn3)
    tet3_conv3 = Conv1D(1, 5, activation='sigmoid', padding='same', name='Convolution_Et3_3')(tet3_dp)
    
    
    
    ########################################## Reconstroction Module ######################################
    
    # Layers of stage 0 contraction
    
    rct0_conv1 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Ct0_1')(tet3_conv3)
    rct0_bn1   = BatchNormalization()(rct0_conv1)
    rct0_conv2 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Ct0_2')(rct0_bn1)
    rct0_bn2   = BatchNormalization()(rct0_conv2)
    rct0_max   = MaxPool1D(pool_size=2, strides=2)(rct0_bn2)
    rct0_dp    = Dropout(0.2)(rct0_max)
    
    # Layers of stage 1 contraction
    
    rct1_conv1 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct1_1')(rct0_dp)
    rct1_bn1   = BatchNormalization()(rct1_conv1)
    rct1_conv2 = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Ct1_2')(rct1_bn1)
    rct1_bn2   = BatchNormalization(rct1_conv2)
    rct1_max   = MaxPool1D(pool_size=2, strides=2)(rct1_bn2)
    rct1_dp    = Dropout(0.2)(rct1_max)
    
    # Layers of stage 2 contraction
    
    rct2_conv1 = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Ct2_1')(rct1_dp)
    rct2_bn1   = BatchNormalization()(rct2_conv1)
    rct2_conv2 = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Ct2_2')(rct2_bn1)
    rct2_bn2   = BatchNormalization()(rct2_conv2)
    rct2_max   = MaxPool1D(pool_size=2, strides=2)(rct2_bn2)
    rct2_dp    = Dropout(0.2)(rct2_max)
    
    # Layers of stage 3 contraction
    
    rct3_conv1 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_1')(rct2_dp)
    rct3_bn1   = BatchNormalization()(rct3_conv1)
    rct3_conv2 = Conv1D(128, 5, activation='relu', padding='same', name='Convolution_Ce3_2')(rct3_bn1)
    rct3_bn2   = BatchNormalization()(rct3_conv2)
    rct3_max   = MaxPool1D(pool_size=2, strides=2)(rct3_bn2)
    rct3_dp    = Dropout(0.2)(rct3_max)
    
    # Layers of stage 1 expansion
    
    ret1_Tconv  = Conv1DTranspose(64, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et1')(rct3_dp)
    ret1_Concat = Concatenate(axis=1)([ret1_Tconv, rct3_conv2])
    ret1_bn1    = BatchNormalization()(ret1_Concat)
    ret1_conv1  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_1')(ret1_bn1)
    ret1_bn2    = BatchNormalization()(ret1_conv1)
    ret1_conv2  = Conv1D(64, 5, activation='relu', padding='same', name='Convolution_Et1_2')(ret1_bn2)
    ret1_dp     = Dropout(0.2)(ret1_conv2)
    
    #Layers of stage 2 expansion
               
    ret2_Tconv  = Conv1DTranspose(32, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et2')(ret1_dp)
    ret2_Concat = Concatenate(axis=1)([ret2_Tconv, rct1_conv2])
    ret2_bn1    = BatchNormalization()(ret2_Concat)
    ret2_conv1  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_1')(ret2_bn1)
    ret2_bn2    = BatchNormalization()(ret2_conv1)
    ret2_conv2  = Conv1D(32, 5, activation='relu', padding='same', name='Convolution_Et2_2')(ret2_bn2)
    ret2_dp     = Dropout(0.2)(ret2_conv2)
                       
    #Layers of stage 3 expansion
               
    ret3_Tconv = Conv1DTranspose(16, 5, strides=2 ,activation='relu', padding='same', name='TransConv_Et3')(ret2_dp)
    ret3_Concat = Concatenate(axis=1)([ret3_Tconv, rce0_conv2])
    ret3_bn1 = BatchNormalization()(ret3_Concat)
    ret3_conv1 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Et3_1')(ret3_bn1)
    ret3_bn2 = BatchNormalization()(ret3_conv1)
    ret3_conv2 = Conv1D(16, 5, activation='relu', padding='same', name='Convolution_Et3_2')(ret3_bn2)
    ret3_bn3 = BatchNormalization()(ret3_conv2)
    ret3_dp = Dropout(0.1)(ret3_bn3)
    ret3_conv2 = Conv1D(1, 5, activation='sigmoid', padding='same', name='Convolution_Et3_3')(ret3_dp)
    
    
    ######################################## Discriminator Module ###############################################
    
    distrib_Y = Input(shape = (2000,20))
    
    ####################################### Transformer Model #################################################3#
    
class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, rate = 0.1):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = layers.TokenAndPositionEmbedding(max_seq_len, input_vocab_size, d_model)
        self.enc_layers = [layers.TransformerEncoderModule(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.max_seq_len = max_seq_len
        self.rate = rate

    def call(self, x, training, mask=None):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x += self.pos_encoding(tf.range(seq_len))
        x = self.dropout(x, training = training)
        for i in range(self.num_layers):
            x, _ = self.enc_layers[i](x, training)

        return x 
        
class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, max_seq_len, rate = 0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = layers.TokenAndPositionEmbedding(max_seq_len, input_vocab_size, d_model)
        self.dec_layers = [layers.TransformerDecoderModule(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.max_seq_len = max_seq_len
        self.rate = rate

    def call(self, x, enc_out, training, mask=None):

        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x += self.pos_encoding(tf.range(seq_len))
        x = self.dropout(x, training = training)
        for i in range(self.num_layers):
            x, weight_1, weight_2 = self.dec_layers[i](x,enc_out, training)
        
            attention_weights['decoder_layer{}_block1'.format(i+1)] = weight_1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = weight_2
    
        return x, attention_weights
    
#class Transformer(Layers):
    