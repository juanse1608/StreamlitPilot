##### Caption Prediction for Images

### Libraries
import streamlit as st
import tensorflow as tf
import numpy as np
import logging
from PIL import Image
import json
import pickle
import matplotlib.pyplot as plt

### Classes

# The CNN encoder used
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # Shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

# The RNN Decoder        
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # Defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # The x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # The x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # Passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # The shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # The x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # Output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

# The BahdanauAttention Class
class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # Features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # Hidden shape == (batch_size, hidden_size)
        # Hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # Attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

        # Score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # Attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class ImageDescriptor():

    def __init__(self):
        pass

    def checkpoint(self, path):

        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt, path, max_to_keep=5)
        epochs = 0
        if ckpt_manager.latest_checkpoint:
            epochs = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            logging.info(f'UPDATING CHECKPOINT WITH {epochs} EPOCHS...')
            # Restoring the latest checkpoint in path
            ckpt.restore(ckpt_manager.latest_checkpoint)
            return epochs

    def parameters(self, path):

        # Parameters
        with open(path) as json_file:
            parameters = json.load(json_file)
        self.parameters = parameters
    
    def architecture(self, path):
        
        # Encoder, decorder and optimizer
        self.encoder = CNN_Encoder(self.parameters['EMBEDDING_DIMENSION'])
        self.decoder = RNN_Decoder(self.parameters['EMBEDDING_DIMENSION'], self.parameters['UNITS'], self.parameters['VOCAB_SIZE'])
        self.optimizer = tf.keras.optimizers.Adam()

        # Feature extractor
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output
        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

        # Tokenizer
        with open(path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
    
    def preprocess(self, path):

        img = tf.image.decode_image(path, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, path
    
    def predict(self, img, type):

        attention_plot = np.zeros((self.parameters['MAX_LENGTH'], self.parameters['ATTENTION_FEATURES_SHAPE']))
        hidden = self.decoder.reset_state(batch_size=1)
        temp_input = tf.expand_dims(img, 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
        features = self.encoder(img_tensor_val)
        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []
        
        for i in range(self.parameters['MAX_LENGTH']):
            predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)
            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            
            # Deterministic prediction (via argmax)
            if type=='Argmax':
                predicted_id = np.argmax(predictions)
            elif type=='Probabilistic':
                predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                result[0] = result[0].title()
                return  result, ' '.join(result[:-1]), attention_plot, i+1
            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        result[0] = result[0].title()
        return result, ' '.join(result[:-1]), attention_plot, i+1
    
    def plot_attention(self, path, result, attention_plot):
        
        temp_image = np.array(tf.image.decode_image(path, channels=3))
        fig = plt.figure(figsize=(20, 20))
        len_result = len(result)
        for i in range(len_result-1):
            temp_att = np.resize(attention_plot[i], (2**3,2**3))
            grid_size = int(max([np.ceil(len_result/2), 4]))
            ax = fig.add_subplot(grid_size, grid_size, i+1)
            ax.set_title(result[i].title())
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.75, extent=img.get_extent())
        plt.tight_layout()
        return fig

