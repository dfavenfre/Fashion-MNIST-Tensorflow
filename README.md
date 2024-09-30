# Fashion-MNIST Neural Network Model Development with Tensorflow
This repository contains a neural network model implemented using TensorFlow for multiclass classification using the Fashion-MNIST dataset. Fashion-MNIST consists of 28x28 grayscale images representing 10 different fashion categories, serving as a replacement for the classic MNIST dataset of handwritten digits. This project demonstrates how to build, train, and evaluate a neural network for image classification tasks.

## Dataset: Fashion-MNIST
The Fashion-MNIST dataset contains:

* Number of Classes: 10 fashion categories (e.g., t-shirts, shoes, bags, etc.)
* Total Images: 70,000 images (60,000 training images and 10,000 test images)
* Image Size: 28x28 pixels (grayscale)
* Sample images from the dataset:
![image](https://github.com/user-attachments/assets/8db7e8dd-1593-4794-a741-ccd381855df9)


# Model Architecture
The model is a fully connected neural network (multilayer perceptron) built using TensorFlow's Sequential API. It includes dropout layers for regularization and L2 regularization for the weights and biases to prevent overfitting.


```Python
model_multiclass = tf.keras.Sequential(
    [
    tf.keras.layers.Flatten(input_shape=(28,28), name="Layer-Flatten-1"),
    tf.keras.layers.Dense(
        units=100,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.LecunNormal(),
        kernel_regularizer=tf.keras.regularizers.L2(),
        bias_initializer=tf.keras.initializers.LecunNormal(),
        bias_regularizer=tf.keras.regularizers.L2(),
        name="Layer-FC_1"
        ),
    tf.keras.layers.Dropout(0.2, name="Layer-Dropout-1"),
    tf.keras.layers.Dense(
        units=100,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.LecunNormal(),
        kernel_regularizer=tf.keras.regularizers.L2(),
        bias_initializer=tf.keras.initializers.LecunNormal(),
        bias_regularizer=tf.keras.regularizers.L2(),
        name="Layer-FC_2"
        ),
    tf.keras.layers.Dropout(0.2, name="Layer-Dropout-2"),
    tf.keras.layers.Dense(
        units=10,
        activation=tf.keras.activations.softmax,
        kernel_initializer=tf.keras.initializers.LecunNormal(),
        kernel_regularizer=tf.keras.regularizers.L2(),
        bias_initializer=tf.keras.initializers.LecunNormal(),
        bias_regularizer=tf.keras.regularizers.L2(),
        name="Layer-Output"
        )
      ], name="Multiclass-Classification-Image-Detection"
)
```

## Model Compilation
The model is compiled using the Adam optimizer and the categorical cross-entropy loss function. The accuracy metric is used to evaluate the model's performance.

```Python
# Compile the model
model_multiclass.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"]

)
```
## Model Training
The model is trained for 100 epochs with the following callbacks for optimization:
* Checkpointing: Save the model during training.
* ReduceLROnPlateau: Reduce learning rate when a metric has stopped improving.
* EarlyStopping: Stop training when a monitored metric stops improving.
* TensorBoard: Log training progress for visualization in TensorBoard.
  
```Python
# Fit and evaluate the model
model_multiclass.fit(
    norm_X_train, tf.one_hot(y_train, depth=10),
    epochs=100,
    validation_data=(norm_X_test, tf.one_hot(y_test, depth=10)),
    callbacks=[
        cb_checkpoint,
        cb_reducelr,
        cb_earlystop,
        tensorboard_callback
        ])
```

# Evaluation

## Confusion Matrix Report
The confusion matrix provides insights into the model's performance across different fashion categories, showing how well the model has classified each category.

![image](https://github.com/user-attachments/assets/65177503-37fb-4a0f-a759-d4e3b71b529b)

## Confusion Matrix With Multiclass Labels
This is a detailed confusion matrix showing the true labels versus predicted labels for all 10 classes.

![image](https://github.com/user-attachments/assets/0e1d642e-f5d7-4aa9-8d1f-f136a4989143)

## Prediction With Unseen Data
The model's performance on new, unseen data is shown below. This demonstrates the model's generalization ability on examples outside the training set.

![image](https://github.com/user-attachments/assets/b6602d6a-90ab-4510-8cc2-643593f1bcaa)


