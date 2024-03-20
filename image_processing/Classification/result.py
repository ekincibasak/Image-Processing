from sklearn.metrics import roc_curve, auc,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle  # Import the pickle module
import numpy as np



def res(model,X_train, X_test, y_train, y_test):

    # Load the training history data from the file
    with open('training_history.pkl', 'rb') as file:
        history_data = pickle.load(file)
        file.close()
        
    # Calculate the mean of the learning curves
    mean_train_loss = np.mean(history_data['train_losses'], axis=0)
    mean_val_loss = np.mean(history_data['val_losses'], axis=0)
    mean_train_accuracy = np.mean(history_data['train_accuracies'], axis=0)
    mean_val_accuracy = np.mean(history_data['val_accuracies'], axis=0)

    # Plot Mean Learning Curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(mean_train_loss, label='Mean Train Loss', color='blue')
    plt.plot(mean_val_loss, label='Mean Validation Loss', color='red')
    plt.legend()
    plt.title('Mean Learning Curves - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(mean_train_accuracy, label='Mean Train Accuracy', color='blue')
    plt.plot(mean_val_accuracy, label='Mean Validation Accuracy', color='red')
    plt.legend()
    plt.title('Mean Learning Curves - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Save the Mean Learning Curves figure
    plt.savefig('mean_learning_curves.png')
    plt.show()



    # Assuming you have trained your model and calculated predictions
    # Get the predicted labels (using the loaded model or your trained model)
    # Here, we assume you've trained the model as before
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

    # Calculate the confusion matrix
    confusion_mtx = confusion_matrix(y_test, y_pred)

    # Calculate normalized confusion matrix
    confusion_mtx_normalized = confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]

    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_mtx_normalized, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")
    plt.title("Normalized Confusion Matrix")
    plt.savefig('Normalized_Confusion_Matrix.png')
    plt.show()



    # Assuming you have trained your model and calculated predictions
    # Get the predicted probabilities for class 1
    y_pred_probs = model.predict(X_test)

    # Calculate the ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)

    # Calculate the AUC (Area Under the Curve)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.savefig('(ROC)_Curve.png')

    plt.legend(loc='lower right')
    plt.show()

    print(f"AUC (Area Under the Curve): {roc_auc:.2f}")


    # Print AUC score
    print(f"AUC (Area Under the ROC Curve): {roc_auc:.2f}")
    from sklearn.metrics import average_precision_score


    # Calculate the Average Precision
    average_precision = average_precision_score(y_test, y_pred_probs)

    # Print Average Precision score
    print(f"Average Precision: {average_precision:.2f}")
    # Print Average Precision score
    print(f"Average Precision: {average_precision:.2f}")

    # Calculate and print precision, recall, and F1-score
    threshold = 0.5  # Choose an appropriate threshold
    y_pred = (y_pred_probs >= threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")