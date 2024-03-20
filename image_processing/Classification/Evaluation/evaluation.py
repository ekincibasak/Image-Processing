import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import itertools
import pickle  # Import the pickle module  # Import the create_model function
from Models import resigmoidel

test_accuracies =[]
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def eva(X_train, X_test, y_train, y_test):
    X_shape = 224 

    # Create your CNN model using the imported function
    #model = create_model()
    model = resigmoidel.Resnet(X_shape)

    # Compile the model with optimizer, loss function, and evaluation metric
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='binary_crossentropy',  # Modified for binary classification
                metrics=['accuracy'])

    # Define the number of folds for cross-validation
    num_folds = 2  # You can adjust this as needed

    # Initialize a KFold object
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Initialize lists for learning curves
    

    # Perform k-fold cross-validation
    for fold, (train_index, val_index) in enumerate(kf.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        print(f"Fold {fold} - Training Data - X_fold_train: {X_fold_train.shape}")
        print(f"Fold {fold} - Training Data - y_fold_train: {y_fold_train.shape}")
        print(f"Fold {fold} - Validation Data - X_fold_val: {X_fold_val.shape}")
        print(f"Fold {fold} - Validation Data - y_fold_val: {y_fold_val.shape}")
        # Train the model on this fold's training data
        history = model.fit(X_fold_train, y_fold_train, epochs=50, batch_size=64, verbose=1, validation_data=(X_fold_val, y_fold_val))
        
        # Learning Curves
        train_losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])
        train_accuracies.append(history.history['accuracy'])
        val_accuracies.append(history.history['val_accuracy'])
        # Evaluate the trained model on the test data (X_test, y_test)
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Fold {fold} - Test Accuracy: {test_accuracy * 100:.2f}%")

        # Store the test accuracy for this fold
        test_accuracies.append(test_accuracy)

        # Store the training history for this fold

    # Calculate and print the average test accuracy across all folds
    avg_test_accuracy = np.mean(test_accuracies)
    print(f"Average Test Accuracy: {avg_test_accuracy * 100:.2f}%")
    # Calculate the mean of the learning curves
    mean_train_loss = np.mean(train_losses, axis=0)
    mean_val_loss = np.mean(val_losses, axis=0)
    mean_train_accuracy = np.mean(train_accuracies, axis=0)
    mean_val_accuracy = np.mean(val_accuracies, axis=0)
    # Save the training history data to a file


    return model


history_data = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'test_accuracies': test_accuracies,
    }

with open('training_history.pkl', 'wb') as file:
    pickle.dump(history_data, file)
    file.close()


