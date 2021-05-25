import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding,Dense,Dropout, Flatten, Input, Concatenate
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D,  Embedding, Reshape

class CNN:
    def __init__(self,config,weight):
        self.config=config
        self.weight=weight
        
    def get_model(self):

        inputs = Input(shape=(self.config.max_length,))
        embedding = Embedding(self.config.vocab_size, 
			                self.config.emb_dim,
                            weights=[self.weight],
                            trainable=True)(inputs)
        
        reshape = Reshape((self.config.max_length,self.config.emb_dim,1))(embedding)
		# CONVOLUTION
        conv_array = []
        maxpool_array = []
        
        for filter in self.config.filter_sizes:
            conv = Conv2D(self.config.nb_filters, (filter, self.config.emb_dim), padding='valid')(reshape)	
            maxpool = MaxPooling2D(pool_size=(self.config.max_length - filter + 1, 1), strides=(1,1), padding='valid')(conv)
            conv_array.append(conv)
            maxpool_array.append(maxpool)			
						
        deconv = Conv2DTranspose(1,(self.config.filter_sizes[0], self.config.emb_dim))(conv_array[0])
        deconv_model = Model(inputs=inputs, outputs=deconv)
        deconv_model.summary()

        if len(self.config.filter_sizes) >= 2:
            merged_tensor = Concatenate()(maxpool_array)
            flatten = Flatten()(merged_tensor)
        else:
            flatten = Flatten()(maxpool_array[0])
		
        dropout = Dropout(self.config.dropout_val)(flatten)
        
        hidden_dense = Dense(self.config.dense_layer_size,kernel_initializer='uniform',activation='relu')(dropout)
        output = Dense(self.config.num_classes, activation='softmax')(hidden_dense)

		# this creates a model that includes
        model = Model(inputs=inputs, outputs=output)
        model.summary()
        
        opt = optimizers.Adam(lr=self.config.lr)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

        return model, deconv_model
	
if __name__=="__main__":
    modelCNN= CNN()
    model,deconv_model=modelCNN.get_model()
