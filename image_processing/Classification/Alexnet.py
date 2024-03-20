
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_model():
    model = Sequential([
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 1)),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=256, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=384, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=384, kernel_size=(3, 3), activation='relu'),
        Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(units=4096, activation='relu'),
        Dropout(0.5),
        Dense(units=4096, activation='relu'),
        Dropout(0.5),
        Dense(units=1, activation='sigmoid')  # Modified for binary classification
    ])

    # Compile the model with optimizer, loss function, and evaluation metric
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',  # Modified for binary classification
                  metrics=['accuracy'])

    return model