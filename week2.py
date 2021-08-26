import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),  #İlk katmanda giriş 28'e 28 düzleştirilmiş bir katmandır (doğrusal dizi yapar)
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  #128 katman var
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  #10 katman var çünkü 10 sınıf var
])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs=5)

model.evaluate(test_images,test_labels)