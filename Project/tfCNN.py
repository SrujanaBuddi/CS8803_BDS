from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

import glob
import numpy as np
import tensorflow as tf
import time

tf.logging.set_verbosity(tf.logging.INFO)
image_size = 28

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 3])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  start_time = time.time()
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  layer1_time = time.time() - start_time
  print ('1:',layer1_time,conv1.shape)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  start_time2 = time.time()
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
  layer2_time = time.time() - start_time2
  print ('2:', layer2_time, conv2.shape) 

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # read data
  train_data_folder = 'Dataset_Augmented/train/Cities'
  classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
  train_images_list = []
  for i in range(len(classes)):
  	
  images_list1 = glob.glob('Dataset_Augmented/train/Cities/*')
  images_list2 = glob.glob('Dataset_Augmented/train/Digits/*')
  trainImages_count = len(images_list1)+len(images_list2)
  train_data = []
  train_labels = []
  for i in range(len(images_list1)):	
  	current_image=Image.open(images_list1[i]);
  	resize_image = current_image.resize((image_size,image_size),Image.ANTIALIAS)
  	image_features = np.concatenate(np.array(resize_image.getdata()))
  	train_data.append(image_features)
  	train_labels.append(0)

  for i in range(len(images_list2)):	
  	current_image=Image.open(images_list2[i]);
  	resize_image = current_image.resize((image_size,image_size),Image.ANTIALIAS)
  	image_features = np.concatenate(np.array(resize_image.getdata()))
  	train_data.append(image_features)
  	train_labels.append(1)

  train_data = np.asarray(train_data, dtype=np.float32)
  train_labels = np.asarray(train_labels, dtype=np.float32)

  # Load training and eval data
  #mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  #train_data = mnist.train.images  # Returns np.array
  #train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/my_models51")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  train_start = time.time()
  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=50,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=100,
      hooks=[logging_hook])
  print ("Training time:%s"%(time.time()-train_start))

  # Predict for unknown images
  predict_images = glob.glob("data/test/*")
  test_data = []
  test_labels = []
  for i in range(len(predict_images)):
  	predict_image = Image.open(predict_images[i])
  	resize_image = predict_image.resize((image_size,image_size),Image.ANTIALIAS)
  	image_features = np.concatenate(np.array(resize_image.getdata()))
	test_data.append(image_features)
  	image_features = np.asarray(image_features, dtype=np.float32)
  	
  test_start = time.time()  
  test_data = np.asarray(test_data, dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
  		x={"x":test_data},
  		num_epochs=1,
  		shuffle=False)
  predicted_results = mnist_classifier.predict(input_fn=predict_input_fn)
  predictions = np.array(list(predicted_results))
  print ('Testing time:%s'%(time.time()-test_start))
  '''for i in range(len(predict_images)):
	print (predict_images[i],predictions[i]['classes'])'''

  '''# Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)'''


if __name__ == "__main__":
  tf.app.run()
