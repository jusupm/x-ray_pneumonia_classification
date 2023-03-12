import tensorflow as tf

# Učitavanje modela
model = tf.keras.models.load_model('my_model.h5')

# Ispis naziva i veličina izlaza svih slojeva
model.summary()