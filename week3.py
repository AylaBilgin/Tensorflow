import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu',
                           input_shape = (28, 28, 1)),    # 64 katmanlı filtre. Filtreler 3'e 3.
    tf.keras.layers.MaxPooling2D(2,2),   # 2x2'lik bir maksimum havuzlama filtresi oluşturulur. Görüntüdeki 4 pikselin en büyük değeri alınır.
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.summary()