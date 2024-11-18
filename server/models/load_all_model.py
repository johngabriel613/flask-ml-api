import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.layers import Layer

# Dictionary to store the loaded models
models = {}

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        self.u = self.add_weight(name="context_vector", shape=(input_shape[-1],), initializer="glorot_uniform", trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        u_it = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        a_it = tf.tensordot(u_it, self.u, axes=1)
        a_it = tf.nn.softmax(a_it)
        output = tf.reduce_sum(x * tf.expand_dims(a_it, -1), axis=1)
        return output
    

# Add other custom objects as needed for exp5
custom_objects = {'Attention': Attention}

def load_all_models():
    model_paths = {
        'exp1': './exp1.keras',
        'exp2': './exp2.keras',
        'exp3': './exp3.keras',
        'exp4': './exp4.keras',
        'exp5': './exp5.keras',
    }
    
    # Load models and print confirmation
    for name, path in model_paths.items():
        try:
            if name == 'exp5':
                # Load exp5 with custom objects
                models[name] = load_model(path, custom_objects=custom_objects)
            else:
                # Load other models normally
                models[name] = load_model(path)
            print(f"Model '{name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading model '{name}': {e}")

# Call the function to load all models
load_all_models()

