
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np

(X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

X_full = np.concatenate([X_train_full, X_test_full], axis=0)
y_full = np.concatenate([y_train_full, y_test_full], axis=0)
total_samples = len(X_full)
train_size = int(0.7 * len(X_full))  # 70% of the total data

# Shuffle the data to ensure randomness
indices = np.random.permutation(len(X_full))
train_indices = indices[:train_size]  # First 70%
test_indices = indices[train_size:]   # Remaining 30%

# Split the data
X_train, y_train = X_full[train_indices], y_full[train_indices]
X_test, y_test = X_full[test_indices], y_full[test_indices]

# Check the shapes of the subsets
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Normalize pixel values
x_train = ((X_train.reshape(-1, 28 * 28).astype('float32') / 255.) - .5) * 2
x_test = ((X_test.reshape(-1, 28 * 28).astype('float32') / 255.) - .5) * 2

# One-hot encode labels - converts integer labels (0-9) into binary vectors
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Define the model
model = Sequential()
# input and first hidden layer
model.add(Dense(input_dim=28*28, units=500))
model.add(Activation('sigmoid'))
# second hidden layer
model.add(Dense(units=500))
model.add(Activation('sigmoid'))
# output layer
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(learning_rate=0.1),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=100, epochs=20)

# Evaluate the model
score = model.evaluate(x_test, y_test)
print('Total loss on Testing Set:', score[0])
print('Accuracy on Testing Set:', score[1])
