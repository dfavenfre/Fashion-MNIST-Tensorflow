def divide_to_sets(
    X: np.array,
    y: np.array,
    val_cut: float,
    test_cut: float) -> tuple :

    """
    Divide the dataset into training, testing, and validation sets.

    Parameters:
    X (numpy.array): Feature values of the dataset.
    y (numpy.array): Target values of the dataset.
    val_cut (float): Proportion of data to be used for validation.
    test_cut (float): Proportion of data to be used for testing.

    Returns:
    tuple: A tuple containing X_train, X_test, y_train, y_test, X_val, y_val.
    """

    validation_size = int(len(X) * val_cut)
    X_val = X[-validation_size:]
    y_val = y[-validation_size:]
    X_train, X_test, y_train, y_test = train_test_split(
        X[:-validation_size],
        y[:-validation_size],
        test_size=test_cut,
        shuffle=True,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, X_val, y_val

def plot_random_image(
    model,
    images,
    true_labels,
    classes):

  i = random.randint(0, len(images))

  target_image = images[i]
  pred_probs = model.predict(target_image.reshape(1,28,28))
  pred_label = classes[pred_probs.argmax()]
  true_label = classes[true_labels[i]]

  plt.imshow(target_image, cmap=plt.cm.binary)

  if pred_label==true_label:
    color="green"
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(
      pred_label,
      100*tf.reduce_max(pred_probs),
      true_label
    ),color=color
  )
  else:
    color ="red"
    plt.xlabel("Pred: {} {:2.0f}% False Prediction / (True: {})".format(
      pred_label,
      100*tf.reduce_max(pred_probs),
      true_label
    ),color=color
  )
