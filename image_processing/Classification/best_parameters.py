import numpy as np
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Assuming you have your data (X_combined_normalized, Y_combined_normalized)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_normalized, Y_combined_normalized, test_size=0.2, random_state=42)

# Define the function to build the model
def build_model(lr=1e-4, epochs=10, batch_size=64):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    
    optimizer = Adam(lr=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

# Initialize a KFold object
num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=False, random_state=42)

# Define a range of learning rates, epochs, and batch sizes to try
learning_rates = [1e-3, 1e-4, 1e-5]
epochs_values = [10, 20, 30]
batch_sizes = [32, 64, 128]

best_lr = None
best_epochs = None
best_batch_size = None
best_accuracy = 0

# Perform a manual search
for lr in learning_rates:
    for epochs in epochs_values:
        for batch_size in batch_sizes:
            accuracies = []
            for train_index, val_index in kf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
                y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]

                model = build_model(lr=lr, epochs=epochs, batch_size=batch_size)
                model.fit(X_fold_train, y_fold_train, epochs=epochs, batch_size=batch_size, verbose=0)

                y_val_pred_proba = model.predict(X_fold_val)
                y_val_pred = np.round(y_val_pred_proba)
                accuracy = accuracy_score(y_fold_val, y_val_pred)
                accuracies.append(accuracy)

            avg_accuracy = np.mean(accuracies)
            print(f"Learning Rate: {lr}, Epochs: {epochs}, Batch Size: {batch_size}, Average Accuracy: {avg_accuracy}")

            if avg_accuracy > best_accuracy:
                best_lr = lr
                best_epochs = epochs
                best_batch_size = batch_size
                best_accuracy = avg_accuracy

# Print the best parameters and corresponding accuracy
print(f"Best Learning Rate: {best_lr}")
print(f"Best Number of Epochs: {best_epochs}")
print(f"Best Batch Size: {best_batch_size}")
print(f"Best Average Accuracy: {best_accuracy}")
