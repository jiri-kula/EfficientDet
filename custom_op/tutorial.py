# %%
import tensorflow as tf
import numpy as np

# Define training dataset and variables
x = [-8, 0.5, 2, 2.2, 201]
y = [-1.4288993, 0.98279375, 1.2490457, 1.2679114, 1.5658458]
offset = tf.Variable(0.0)


# Define a simple model which just contains a custom operator named `Atan`
@tf.function(input_signature=[tf.TensorSpec.from_tensor(tf.constant(x))])
def atan(x):
    return tf.atan(x + offset, name="Atan")


@tf.function(input_signature=[tf.TensorSpec.from_tensor(tf.constant(x))])
def model(x):
    return y + atan(x)


# Train model
optimizer = tf.optimizers.Adam(0.01)


def train(x, y):
    with tf.GradientTape() as t:
        predicted_y = model(x)
        loss = tf.reduce_sum(tf.square(predicted_y - y))
    grads = t.gradient(loss, [offset])
    optimizer.apply_gradients(zip(grads, [offset]))


for i in range(1000):
    train(x, y)

print("The actual offset is: 1.0")
print("The predicted offset is:", offset.numpy())

# %% Convert to tflite
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [model.get_concrete_function()], model
)

converter.allow_custom_ops = True
tflite_model = converter.convert()

# %% Save the model.
with open("atan.tflite", "wb") as f:
    f.write(tflite_model)

# %%
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# %%
