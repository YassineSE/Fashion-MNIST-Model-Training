import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
from PIL import Image,ImageOps

model = tf.keras.models.load_model("./fashion_mnist_conv.h5")

(train_data, train_labels) , (test_data, test_labels) = fashion_mnist.load_data()
train_data , test_data = train_data.astype("float32")/255.0, test_data.astype("float32")/255.0

#print(model.predict(train_data[0].reshape(1,28,28)))

classes = [
    "t-shirt",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle boot"
]
'''
def plot_random_image(model, images, true_labels, classes):
  """
  Pick a random image, plots it and labels it with a prediction and truth label
  """

  # Set up random integer
  i = random.randint(0, len(images))

  # Create predictions and targets
  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1,28,28))
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  # Plot image
  plt.imshow(target_image, cmap=plt.cm.binary)

  #Change the color of the titles depending on the prediction
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  # Add xlabel information
  plt.xlabel(f"Pred: {pred_label} {100*tf.reduce_max(pred_probs):2.0f}% (True: {true_label})", color=color)
  
  plt.show()

plot_random_image(model=model, images=test_data, true_labels=test_labels, classes=classes)
'''
test_image = np.array(ImageOps.invert(Image.open("./test_tshirt.jpg.jpg").convert("L")))

pred_probs = model.predict(test_image.reshape(1,28,28))
pred_label = classes[pred_probs.argmax()]
plt.imshow(test_image, cmap=plt.cm.binary)
print(pred_label, (100*tf.reduce_max(pred_probs)).numpy())
plt.show()