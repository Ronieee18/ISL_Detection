import tensorflow as tf

# Load your Keras model from .h5 file
model = tf.keras.models.load_model('C:/Users/RONIT/Desktop/SIH/isl_model.h5')

# Save the model as a SavedModel
model.save('C:/Users/RONIT/Desktop/SIH/saved_model')
