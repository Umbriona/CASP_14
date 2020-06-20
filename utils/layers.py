import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Conv2DTranspose, Lambda, Dense, LayerNormalization, Dropout, Concatenate, Conv1D, MaxPool1D, BatchNormalization, UpSampling1D
from tensorflow.keras import Sequential
from tensorflow.keras.regularizers import L1L2
import tensorflow.keras.backend as K

######################################### U-Net components #############################################

class Conv1DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, activation = 'relu', name=None, *args, **kwargs):
        self._filters = filters
        self._kernel_size = (1, kernel_size)
        self._strides = (1, strides)
        self.activation = activation
        self._args, self._kwargs = args, kwargs
        
        super(Conv1DTranspose, self).__init__(name=name)

    def build(self, input_shape):
        print("build", input_shape)
        self._model = Sequential()
        self._model.add(Lambda(lambda x: K.expand_dims(x,axis=1), batch_input_shape=input_shape))
        self._model.add(Conv2DTranspose(self._filters,
                                        kernel_size=self._kernel_size,
                                        strides=self._strides,
                                        activation = self.activation,
                                        padding='same',
                                        *self._args, **self._kwargs))
        self._model.add(Lambda(lambda x: x[:,0]))
        #self._model.summary()
        #super(Conv1DTranspose, self).build(input_shape)

    def call(self, x):
        return self._model(x)

    def compute_output_shape(self, input_shape):
        return self._model.compute_output_shape(input_shape)
    
class DownSampleMod(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_max_pool=False,
                 pool_size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01):
        super(DownSampleMod, self).__init__()
        self.use_max_pool = use_max_pool
        self.bn1   = BatchNormalization()
        self.bn2   = BatchNormalization()
        self.bn3   = BatchNormalization()

        self.reg1 = L1L2(l1=l1, l2=l2)
        self.reg2 = L1L2(l1=l1, l2=l2)
        self.reg3 = L1L2(l1=l1, l2=l2)

        self.conv1 = Conv1D(num_filter, size_filter, activation='relu', padding='same',
                            kernel_regularizer = self.reg1, bias_regularizer = self.reg1)
        self.conv2 = Conv1D(num_filter, size_filter, activation='relu', padding='same',
                            kernel_regularizer = self.reg2, bias_regularizer = self.reg2)
        if not self.use_max_pool:
            self.d_sample = Conv1D(num_filter, size_filter, activation='relu', padding='same', strides = sampling_stride,
                            kernel_regularizer = self.reg3, bias_regularizer = self.reg3)
        else:
            self.d_sample = MaxPool1D(pool_size = pool_size, strides = sampling_stride)
        self.dOut = Dropout(rate)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        out_enc = self.bn2(x)
        x = self.d_sample(out_enc)
        out = self.bn3(x)
        if not self.use_max_pool:
            return self.dOut(out), self.dOut(out_enc)
        else:
            return self.dOut(out), self.dOut(out_enc)

        
class UpsampleMod(Layer):
    def __init__(self, num_filter, size_filter, sampling_stride, use_regular_uppsampling = False,
                size = 2, rate = 0.1, l1 = 0.01, l2 = 0.01, use_max_pool = False):
        super(UpsampleMod, self).__init__()
        self.use_max_pool = use_max_pool
        self.bn1   = BatchNormalization()
        self.bn2   = BatchNormalization()
        self.bn3   = BatchNormalization()

        self.reg1 = L1L2(l1=l1, l2=l2)
        self.reg2 = L1L2(l1=l1, l2=l2)
        self.reg3 = L1L2(l1=l1, l2=l2)
        
        self.concat = Concatenate(axis=2)

        self.conv1 = Conv1D(num_filter, size_filter, activation='relu', padding='same',
                            kernel_regularizer = self.reg1, bias_regularizer = self.reg1)
        self.conv2 = Conv1D(num_filter, size_filter, activation='relu', padding='same',
                            kernel_regularizer = self.reg2, bias_regularizer = self.reg2)
        if not self.use_max_pool:
            self.u_sample = Conv1DTranspose(num_filter, size_filter, activation='relu', strides = sampling_stride,
                            kernel_regularizer = self.reg3, bias_regularizer = self.reg3)
        else:
            self.u_sample = UpSampling1D(size = size)
        self.dOut = Dropout(rate)
    
    def call(self, x, enc):
        x = self.u_sample(x)
        x = self.concat([x, enc])
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.dOut(x) 
        return x
    
class UNetModule(Layer):
    def __init__(self, p):
        super(UNetModule, self).__init__()
        self.num_filter = p['num_filter']
        self.size_filter = p['kernel_size']
        self.sampling_stride = p['sampling_stride']
        self.sampling_stride.append(1)
        self.use_max_pool = p['use_max_pool']
        self.pool_size = p['pool_size']
        self.rate = p['rate']
        self.l1 = p['l1']
        self.l2 = p['l2']
        
        self.comp_stage = [DownSampleMod(self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                            use_max_pool=False, pool_size = self.pool_size[i], rate = self.rate[i],
                                            l1 = self.l1[i], l2 = self.l2[i]) for i in range(4)]
        self.exp_stage = [UpsampleMod( self.num_filter[i], self.size_filter[i], self.sampling_stride[i],
                                           use_regular_uppsampling = False, size = self.sampling_stride[i],
                                           rate = self.rate[i], l1 = self.l1[i], l2 = self.l2[i])  for i in range(3)]
          

    def call(self, x):
        tmp = []
        for i in range(4):
            x, out_enc = self.comp_stage[i](x)
            tmp.append(out_enc) 

        for i in reversed(range(3)):
            x = self.exp_stage[i](x, tmp[i])
        return(x)
    
##################################### Transformer Components ############################################
    
class MultiHeadedAttention(Layer):
    def __init__(self, input_dim, num_heads = 1, name = None, *args, **kwargs):
        super(MultiHeadedAttention, self).__init__(name=name)
        self.input_dim = input_dim
        self.heads = num_heads
        if self.input_dim % self.n_heads != 0:
            raise ValueError("input_dim should be divisable num_heads")
        self.proj_dim = self.input_dim // self.num_heads
        self.query_proj = Dense(self.proj_dim)
        self.key_proj = Dense(self.proj_dim)
        self.value_proj = Dense(self.proj_dim)
        self.concat_proj = Dense(self.input_dim)
        
#    def build(self, input_shape):
        
#        self._model = 
    def attention(self, query, key, value):
        attend = tf.matmul(query, key, transpose_b=True)
        scaled_attend = attend / tf.math.sqrt(self.proj_dim)
        weights = tf.nn.softmax(scaled_attend)
        output = tf.matmul(weights,value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x,(batch_size,-1, self.num_heads, self.proj_dim))
        return tf.transpose(x, perm=[0,2,1,3])
    
    def call(self, x1, x2, x3):
        batch_size = tf.shape(x)[0]
        query =self.query_dens(x1)
        key = self.key_dens(x2)
        value = self.value_dense(x3)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output, weights

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
    
class TransformerEncoderModule(Layer):
    def __init__(self, input_dim, num_heads, ff_dim, d_rate=0.1):
        super(TransformerEncoderModule, self).__init__()
        self.attend = MultiHeadedAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=True):
        attn_output, weights_enc = self.attend(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
class TransformerDencoderModule(Layer):
    def __init__(self, input_dim, num_heads, ff_dim, d_rate=0.1):
        super(TransformerDecoderModule, self).__init__()
        self.attend_1 = MultiHeadedAttention(embed_dim, num_heads)
        self.attend_2 = MultiHeadedAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)
        
    def call(self, target, enc_out, training=True):
        attn_output, weights_dec_1 = self.attend(target, target, target)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(target + attn_output)
        
        attn_output, weights_dec_2 = self.attend(out1, enc_out, enc_out)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = self.layernorm2(target + attn_output)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)
    