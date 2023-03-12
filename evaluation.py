import tensorflow as tf


model = tf.keras.models.load_model('my_model.h5')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'dataset/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

predictions = model.predict(test_generator)
true_classes = test_generator.classes
predicted_classes = tf.argmax(predictions, axis=1)

accuracy = tf.keras.metrics.Accuracy()(true_classes, predicted_classes)
precision = tf.keras.metrics.Precision()(true_classes, predicted_classes)
recall = tf.keras.metrics.Recall()(true_classes, predicted_classes)

print("Accuracy:", accuracy.numpy())
print("Precision:", precision.numpy())
print("Recall:", recall.numpy())