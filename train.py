import tensorflow as tf
import os
import json

# Get the output path from the Valohai machines environment variables
output_path = os.getenv('VH_OUTPUTS_DIR', '.outputs/')



def logMetadata(epoch,logs):
            print()
            print(json.dumps({
                        'epoch':epoch,
                        'loss':str(logs['loss']),
                        'acc':str(logs['accuracy']),
            }))

metadataCallback = tf.keras.callbacks.LambdaCallback(on_epoch_end=logMetadata)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.save(os.path.join(output_path, 'model.h5'))

