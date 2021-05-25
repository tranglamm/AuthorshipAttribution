import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class Attention:
    def build(config):
        input_layer = Input(shape=(config.max_length,),dtype='int32', name='main_input')
        emb_layer = Embedding(config.vocab_size,
                            config.emb_dim,
                            input_length=config.max_length)(input_layer)

        ####GRU bidirectionnal
        bi_GRU_layer = Bidirectional(GRU(config.rnn_dim,return_sequences=True,reset_after=False))(emb_layer)

        #Attention
        ####
        uit_layer = TimeDistributed(Dense(config.rnn_dim*2,use_bias=True,activation='tanh'))(bi_GRU_layer)
        uw_dense = Dense(1,input_shape=(config.rnn_dim*2,))


            
        uitw_layer = TimeDistributed(uw_dense)(uit_layer)

        def exponential(v):
            #return K.tf.exp(v)	
            return K.exp(v)
        uitw_layer = Lambda(exponential,output_shape=(config.max_length,1))(uitw_layer)

        def sum(v):
            #return K.tf.reduce_sum(v,1)
            return tf.math.reduce_sum(v,1)

        sum_uitw_layer = Lambda(sum,output_shape=(1,))(uitw_layer)

        sum_uitw_layer = RepeatVector(config.max_length)(sum_uitw_layer)

        def attention_compute(v):
            up_part = v[0]
            low_part = v[1]
            #return K.tf.divide(up_part,low_part)
            return tf.math.divide(up_part,low_part)
        ait_layer = Lambda(attention_compute,output_shape=(config.max_length,),name="attention_compute")([uitw_layer,sum_uitw_layer])


        #End attention

        def apply_attention(v):
            ait = v[0]
            hit = v[1]	
            return tf.math.reduce_sum(tf.math.multiply(hit,ait),1)

        repeat_ait_layer = Flatten()(ait_layer)
        repeat_ait_layer = RepeatVector(config.rnn_dim*2)(repeat_ait_layer)

        repeat_ait_layer = Permute((2, 1))(repeat_ait_layer)	

        attention_bi_GRU_layer = Lambda(apply_attention,output_shape=(config.rnn_dim*2,),name="apply_attention")([repeat_ait_layer,bi_GRU_layer])

        output_layer = Dense(config.num_classes,activation="softmax")(attention_bi_GRU_layer)

        model = Model(inputs=[input_layer], outputs=[output_layer])
        attention_model = Model(inputs=[input_layer],outputs=[ait_layer])

        model.summary()
        attention_model.summary()
        return model,attention_model
        