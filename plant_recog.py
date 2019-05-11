#%%
import tensorflow.keras.backend as K
!pip install -q tensorflow_hub
#%%
from __future__ import absolute_import, division, print_function

import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image

from tensorflow.keras import layers

#%%
data_root = tf.keras.utils.get_file(
    'flower_photos', 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
#%%

# The simplest way to load this data into our model is using tf.keras.preprocessing.image.ImageDataGenerator:

# All of TensorFlow Hub's image modules expect float inputs in the [0, 1] range.
# Use the ImageDataGenerator's rescale parameter to achieve this.

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root))

#%%
# Returns image_batch, label_batch pairs.
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

#%%
# Use hub.module to load a mobilenet, 
# and tf.keras.layers.Lambda to wrap it up as a keras layer.
classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"

def classifier(x):
  classifier_module = hub.Module(classifier_url)
  return classifier_module(x)

IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))

#%%
classifier_layer = layers.Lambda(classifier, input_shape=IMAGE_SIZE+[3])
classifier_model = tf.keras.Sequential([classifier_layer])
classifier_model.summary()
# Outputs:
# INFO: tensorflow: Saver not created because there are no variables in the graph to restore

#%%
# Rebuild the data generator, 
# with the output size set to match what's expected by the module.
image_data = image_generator.flow_from_directory(
    str(data_root), target_size=IMAGE_SIZE)
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break

#%%
# Manually initializing TFHub modules
# import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)
#%% 
# Test run on a single image
# import numpy as np
# import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SIZE)
grace_hopper 
#%%
grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape
# Add a batch dimension
# The result is a 1001 element vector of logits, 
# rating the probability of each class for the image.
result = classifier_model.predict(grace_hopper[np.newaxis, ...])
result.shape

#%%
# Find the top class ID with argmax
predicted_class = np.argmax(result[0], axis=-1)
predicted_class
#%%
# Use predicted class ID to fetch the ImageNet labels
# and decode the predictions
labels_path = tf.keras.utils.get_file(
    'ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
#%%
plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name)

#%%
# Run in on a batch
result_batch = classifier_model.predict(image_batch)
labels_batch = imagenet_labels[np.argmax(result_batch, axis=-1)]
labels_batch
#%%
# Check
plt.figure(figsize=(10, 9))
for n in range(30):
  plt.subplot(6, 5, n+1)
  plt.imshow(image_batch[n])
  plt.title(labels_batch[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

#%%
# Simple Transfer Learning
# Using tfhub to retrain the top layer of the model to recognize the classes in our dataset.

# Download headless model
# @param {type:"string"}
feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"
#%%
# Create module and check size
def feature_extractor(x):
  feature_extractor_module = hub.Module(feature_extractor_url)
  return feature_extractor_module(x)


IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))

#%%
image_data = image_generator.flow_from_directory(
    str(data_root), target_size=IMAGE_SIZE)
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Labe batch shape: ", label_batch.shape)
  break


#%%
# Wrap module in keras layer
features_extractor_layer = layers.Lambda(
    feature_extractor, input_shape=IMAGE_SIZE+[3])


#%%
# Freeze variable
# So training only modifies the new classifier layer
features_extractor_layer.trainable = False


#%%
# Attach a classifer head:
# Wrap the hub layer in a tf.keras.Sequential model, 
# and add a new classification layer.

model = tf.keras.Sequential([
    features_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()


#%%
# Initialize TFHub
init = tf.global_variables_initializer()
sess.run(init)


#%%
# Test run a single batch
result = model.predict(image_batch)
result.shape

#%%
# Train the model using compile configure the training process:
model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#%%
# Use the .fit method to train the model.


class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])

# Using 1 epoch
steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=10,
          steps_per_epoch=steps_per_epoch,
          callbacks=[batch_stats])


#%%
# Check progress
plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(batch_stats.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(batch_stats.batch_acc)

#%%
# Check predictions
# Get ordered list of class names:
label_names = sorted(image_data.class_indices.items(),
                     key=lambda pair: pair[1])
label_names = np.array([key.title() for key, value in label_names])
label_names

#%%
# Run the image batch through the model and convert the indices to class names.
result_batch = model.predict(image_batch)
labels_batch = label_names[np.argmax(result_batch, axis=-1)]
labels_batch

# Plot the result
plt.figure(figsize=(10, 9))
for n in range(30):
  plt.subplot(6, 5, n+1)
  plt.imshow(image_batch[n])
  plt.title(labels_batch[n])
  plt.axis('off')
_ = plt.suptitle("Model predictions")


#%%
# Export model
export_path = tf.contrib.saved_model.save_keras_model(model, "./saved_models")
export_path




#%%
# Convert model to tflite
converter = tf.lite.TFLiteConverter.from_saved_model(
    "./saved_models\\1557539094")
print("Succeeded")
tflite_model = converter.convert()
print("Opening")
open("converted_model.tflite", "wb").write(tflite_model)


#%%
