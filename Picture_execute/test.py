import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.enable_eager_execution()

print("TensorFlow Version:\t", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

fig, ax = plt.subplots(nrows=5, ncols=5, sharex='all', sharey='all')
ax = ax.flatten()
for i in range(25):
    img = x_train[i].reshape(28, 28)
    ax[i].set_title(y_train[i])
    # ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].imshow(img, cmap='gray_r', interpolation='nearest')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()