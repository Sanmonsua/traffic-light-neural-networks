# Modeling workflow

### Attempt 1: Barebones

First setup is going to be a single 3D Input layer and an Output layer of size NUM_CATEGORIES

- The results are quite interesting, as is every training run returns a very random accuracy, so we are not very consistent (small dataset)
- On the large dataset we can't even get a decent result.

accuracy: 0.0042 - loss: 28.6449 👎

```python
model = keras.Sequential([
    keras.Input((IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(NUM_CATEGORIES, 'sigmoid'),
])
```

### Attempt 2: Convolutional Layer

Second attempt, clearly our neural network is not able to identify features or build any relationship with the image and the labels. This is why we are now going to introduce a convolutional layer.

- Starting with 32 filters
- 3x3 kernel
- relu activation function

accuracy: 0.8763 - loss: 1.2651

Significant improvements. 😃

```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(NUM_CATEGORIES, 'sigmoid'),
])
```

### Attempt 2.1

Let's expirement with above parameters to see if we see any improvements. Starting with using 64 filters instead:

accuracy: 0.8710 - loss: 1.1599 -> No improvements

```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(NUM_CATEGORIES, 'sigmoid'),
])
```

### Attempt 2.2

Using 5x5 kernel.
accuracy: 0.8695 - loss: 1.0422 -> No improvements

```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.Flatten(),
    keras.layers.Dense(NUM_CATEGORIES, 'sigmoid'),
])
```

### Attempt 2.3

Using LeakyReLU activation function
accuracy: 0.0273 - loss: 5.9528 -> 

Ridiculous 😆

```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(NUM_CATEGORIES, 'sigmoid'),
])
```

### Attmept 3: Pooling
Definitely the magic came once we introduced our convolutional layer, but another important technique we learned to improve performance.

accuracy: 0.9176 - loss: 0.6412 😃

```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(NUM_CATEGORIES, 'sigmoid'),
])
```

### Attmept 4: Repeating
After several tries and attempts, I started playing with adding extra convulitional + pooling layers, and I found accuracy greatly improved as I added extra layers + filters.

accuracy: 0.9580 - loss: 0.2348 🔥
```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(NUM_CATEGORIES, 'softmax'),
])
```

### Attmept 4: Hidden Layer
Ok so now is time to make our model more powerful, this time I tried adding a hidden layer alone of size 128, which turn out somewhat good.

accuracy: 0.9644 - loss: 0.1796 🆙
```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(128, 'relu'),

    keras.layers.Dense(NUM_CATEGORIES, 'softmax'),
])
```
#### Attempt 4.1: More units
Even though we see some improvements from adding extra units, it doesn't seem as we make much progress, considering that 

accuracy: 0.9623 - loss: 0.1795 😐
```python
model = keras.Sequential([
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
        keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),

        keras.layers.Flatten(),

        keras.layers.Dense(256, 'relu'),

        keras.layers.Dense(NUM_CATEGORIES, 'softmax'),
    ])
```

### Attempt 5: Regularization 
One of the areas our model may be struggling now is that is overfitting too much on the training data so it doesn't perform as good on the test data because we don't get to generalize on the problem.

accuracy: 0.9800 - loss: 0.0777 ❤️‍🔥

```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(256, 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, 'relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(NUM_CATEGORIES, 'softmax'),
])
```

#### Attempt 5.1: More layers
Adding an extra layer, yet not good 😅

accuracy: 0.9214 - loss: 0.2726

```python
model = keras.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D((2,2)),

    keras.layers.Flatten(),

    keras.layers.Dense(256, 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(256, 'relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(NUM_CATEGORIES, 'softmax'),
])
```