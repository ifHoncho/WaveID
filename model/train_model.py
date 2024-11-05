#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import keras
import keras.backend as K
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow.keras as tk

# Constants
DATASET_FILE_PATH = "dataset/GOLD_XYZ_OSC.0001_1024.hdf5"
BASE_MODULATION_CLASSES = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK',
                           '16APSK', '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM',
                           '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC',
                           'FM', 'GMSK', 'OQPSK']
SELECTED_MODULATION_CLASSES = ['4ASK', 'BPSK', 'QPSK', '16PSK', '16QAM', 'FM', 'AM-DSB-WC', '32APSK']
N_SNR = 4  # from 30 SNR to 22 SNR


# In[2]:


# Load Data
def load_data(dataset_file_path, selected_classes, n_snr):
    # Load the dataset from the given HDF5 file path
    dataset_file = h5py.File(dataset_file_path, "r")
    # Identify the indices for the selected modulation classes
    selected_classes_id = [BASE_MODULATION_CLASSES.index(cls) for cls in selected_classes]

    X_data, y_data = None, None

    # Loop through each selected class ID and extract corresponding data slices
    for idx in selected_classes_id:
        X_slice = dataset_file['X'][(106496 * (idx + 1) - 4096 * n_snr):106496 * (idx + 1)]
        y_slice = dataset_file['Y'][(106496 * (idx + 1) - 4096 * n_snr):106496 * (idx + 1)]

        # Concatenate data slices for all selected classes
        if X_data is not None:
            X_data = np.concatenate([X_data, X_slice], axis=0)
            y_data = np.concatenate([y_data, y_slice], axis=0)
        else:
            X_data, y_data = X_slice, y_slice

    # Reshape data to match the model input requirements (32x32 with 2 channels)
    X_data = X_data.reshape(len(X_data), 32, 32, 2)
    y_data_df = pd.DataFrame(y_data)

    # Remove columns with all zeros, indicating no presence of those classes
    for column in y_data_df.columns:
        if sum(y_data_df[column]) == 0:
            y_data_df = y_data_df.drop(columns=[column])

    # Rename columns to match the selected modulation classes
    y_data_df.columns = selected_classes
    return X_data, y_data_df


# In[3]:


# Load the dataset
X_data, y_data = load_data(DATASET_FILE_PATH, SELECTED_MODULATION_CLASSES, N_SNR)


# In[4]:


# Split Data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)


# In[5]:


# Model Creation
def create_model(input_shape=(32, 32, 1), learning_rate=0.0001):
    # Define separate input layers for I and Q components
    i_input = keras.layers.Input(shape=input_shape)
    q_input = keras.layers.Input(shape=input_shape)

    # Define a branch for CNN processing
    def cnn_branch(input_layer):
        # First convolutional layer with LeakyReLU activation
        cnn_1 = tk.layers.Conv2D(64, 3, activation=LeakyReLU(alpha=0.1))(input_layer)
        # Second convolutional layer with LeakyReLU activation
        cnn_2 = tk.layers.Conv2D(64, 3, activation=LeakyReLU(alpha=0.1))(cnn_1)
        # Max pooling layer to reduce spatial dimensions
        pool_1 = tk.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(cnn_2)
        # Third convolutional layer
        cnn_3 = tk.layers.Conv2D(128, 3, activation=LeakyReLU(alpha=0.1))(pool_1)
        # Fourth convolutional layer
        cnn_4 = tk.layers.Conv2D(128, 3, activation=LeakyReLU(alpha=0.1))(cnn_3)
        # Another max pooling layer
        pool_2 = tk.layers.MaxPool2D(pool_size=3, strides=2, padding='valid')(cnn_4)
        # Flatten the output for use in fully connected layers
        return tk.layers.Flatten()(pool_2)

    # Apply the CNN branch to both I and Q inputs
    flatten_q = cnn_branch(q_input)
    flatten_i = cnn_branch(i_input)
    
    # Concatenate the flattened outputs from both branches
    concat = keras.layers.concatenate([flatten_q, flatten_i])
    
    # Fully connected layer with LeakyReLU activation
    dense1 = keras.layers.Dense(1024, activation=LeakyReLU(alpha=0.1))(concat)
    # Dropout layer for regularization
    dropout1 = tk.layers.Dropout(0.5)(dense1)
    # Another fully connected layer
    dense2 = keras.layers.Dense(256, activation=LeakyReLU(alpha=0.1))(dropout1)
    # Output layer with softmax activation for classification
    outputs = keras.layers.Dense(len(SELECTED_MODULATION_CLASSES), activation='softmax')(dense2)

    # Define the model with I and Q inputs
    model = keras.Model(inputs=[i_input, q_input], outputs=outputs)
    # Compile the model with categorical crossentropy loss and Adam optimizer
    model.compile(
        loss='categorical_crossentropy', 
        optimizer=tk.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model


# In[6]:


# Create the model
model = create_model()


# In[7]:


# Training Function
def train_model(model, X_train, y_train, X_val, y_val, batch_size=64, epochs=25):
    # Path for saving the best model checkpoint
    path_checkpoint = "model/model_checkpoint.weights.h5"
    # Early stopping callback to stop training if accuracy does not improve
    es_callback = EarlyStopping(monitor="accuracy", min_delta=0, patience=10)

    # Model checkpoint callback to save the best model during training
    modelckpt_callback = ModelCheckpoint(
        monitor="accuracy",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=True,
        save_best_only=True,
    )

    # Train the model using training data and validation data
    history = model.fit(
        x=[X_train[:, :, :, 0], X_train[:, :, :, 1]],
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_val[:, :, :, 0], X_val[:, :, :, 1]], y_val),
        callbacks=[es_callback, modelckpt_callback],
    )
    return history


# In[8]:


# Train the model
history = train_model(model, X_train, y_train, X_test, y_test)


# In[9]:


# Plotting Function
def plot_training_history(history):
    # Plot training and validation accuracy over epochs
    plt.plot(np.c_[history.history['accuracy'], history.history['val_accuracy']])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.show()

    # Plot training and validation loss over epochs
    plt.plot(np.c_[history.history['loss'], history.history['val_loss']])
    plt.legend(['loss', 'val_loss'])
    plt.show()


# In[10]:


# Plot Training History
plot_training_history(history)


# In[15]:


# Predict using the trained model
model_predictions = model.predict([X_test[:, :, :, 0], X_test[:, :, :, 1]])

# Convert the logits to class indices
def convert_to_matrix(logit_list):
    logit_list = list(logit_list)
    return logit_list.index(max(logit_list))

# Compute the confusion matrix
cm = confusion_matrix(
    y_true=list(map(convert_to_matrix, y_test.values)),
    y_pred=list(map(convert_to_matrix, model_predictions)),
)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=SELECTED_MODULATION_CLASSES)
disp.plot()
plt.show()

# Print classification report for precision, recall, and F1-score
print(classification_report(list(map(convert_to_matrix, y_test.values)),
                                list(map(convert_to_matrix, model_predictions)),
                                target_names=SELECTED_MODULATION_CLASSES))


# In[13]:


# Save the trained model to a file using Keras's built-in method
model_filename = "model/trained_model.h5"
model.save(model_filename)

print(f"Model saved as {model_filename}")


# In[14]:


model.summary()  # This will show you the model's layers, including input shapes
print(model.input)  # This will give more details about the input layer

