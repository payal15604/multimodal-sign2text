#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[3]:


get_ipython().system('pip install ipykernel')


# In[2]:


get_ipython().system('pip install numpy pandas matplotlib seaborn plotly pillow scikit-learn optree')


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import plot_model

import warnings
warnings.filterwarnings("ignore")


# In[8]:


train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")


# In[9]:


fig = px.histogram(train_df, x='label',color='label', title='Distribution of Labels in Training Dataset')

fig.update_layout(
    xaxis_title='Label',
    yaxis_title='Count',
    showlegend=False,
    bargroupgap=0.1,
)

fig.show()


# In[10]:


# Group the dataframe by the 'label' column
label_groups = train_df.groupby('label')

# Iterate over each label group and display one image
fig, axs = plt.subplots(4, 6, figsize=(12, 8))

for i, (label, group) in enumerate(label_groups):
    # Get the first image from the group
    image = group.iloc[0, 1:].values.reshape(28, 28)
    
    # Calculate the subplot index
    row = i // 6
    col = i % 6
    
    # Convert label to integer and add 65 to get ASCII value
    ascii_value = int(label) + 65
    
    # Display the image
    axs[row, col].imshow(image, cmap='gray')
    axs[row, col].set_title(chr(ascii_value))
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()


# In[7]:


X_train = train_df.drop(labels = ["label"],axis = 1) 
y_train = train_df["label"]


# In[8]:


X_test = test_df.drop(labels = ["label"],axis = 1)
y_test = test_df["label"]


# In[9]:


X_train = np.array(X_train, dtype='float32')
X_test = np.array(X_test, dtype='float32')
y_train = np.array(y_train, dtype='float32')
y_test = np.array(y_test, dtype='float32')


# In[10]:


X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


# In[11]:


X_train = X_train / 255.0
X_test = X_test / 255.0


# In[12]:


num_classes = 25
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)


# In[13]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

# Define the number of classes (make sure it's set correctly)
num_classes = 25  # Update based on your dataset

# Build the Model
model = Sequential()

# First Convolutional Block
model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second Convolutional Block
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))  # Regularization to prevent overfitting
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the Model
model.compile(
    optimizer=RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Model Summary (Optional)
model.summary()


# In[32]:


get_ipython().system('pip install pydot graphviz')


# In[33]:


from tensorflow.keras.utils import plot_model
plot_model(model, to_file='/kaggle/working/model_architecture1.png', show_shapes=True, show_layer_names=True)


# In[34]:


get_ipython().system('pip install graphviz')


# In[35]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)


# In[36]:


augmented_images = []
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9):
    for img in X_batch:
        augmented_images.append(img)
    break  

# Display augmented images
plt.figure(figsize=(10, 10))
for i, image in enumerate(augmented_images):
    plt.subplot(3, 3, i + 1)
    plt.imshow(image.squeeze(), cmap='gray')  # Squeeze to remove the channel dimension
    plt.title(f'Augmented Image {i + 1}', fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.show()


# In[37]:


learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3,factor=0.5, min_lr=0.0001)

history = model.fit(datagen.flow(X_train, y_train), epochs=25, validation_data=(X_test, y_test), verbose=1, callbacks=[learning_rate_reduction])


# In[38]:


# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Define plot labels and styles
plot_labels = ['Accuracy', 'Loss']
plot_styles = ['-', '--']

# Plot training and testing accuracy/loss
for i, metric in enumerate(['accuracy', 'loss']):
    train_metric = history.history[metric]
    test_metric = history.history['val_' + metric]
    axs[i].plot(train_metric, label='Training ' + metric.capitalize(), linestyle=plot_styles[0])
    axs[i].plot(test_metric, label='Testing ' + metric.capitalize(), linestyle=plot_styles[1])
    axs[i].set_xlabel('Epochs')
    axs[i].set_ylabel(plot_labels[i])
    axs[i].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()


# In[39]:


# Get the model's predictions on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)


# In[40]:


cm = confusion_matrix(y_true_classes, y_pred_classes)


# In[41]:


# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[42]:


accuracy = model.evaluate(X_test, y_test)
print(f'validation test results - Loss: {accuracy[0]} - Accuracy: {accuracy[1]*100}%')


# In[44]:


model.save("americanSignLanguage.keras")


# In[45]:


# Get model predictions for the first 10 images in the test set
predictions_asl = model.predict(X_test[:10])
predicted_labels_asl = np.argmax(predictions_asl, axis=1)
actual_labels_asl = np.argmax(y_test[:10], axis=1)

# Decode labels using ASCII values
predicted_labels_asl = [chr(label + 65) for label in predicted_labels_asl]
actual_labels_asl = [chr(label + 65) for label in actual_labels_asl]

# Display actual vs predicted images for the first 10 images
plt.figure(figsize=(15, 7))
for i in range(10):
    # Display actual image
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {actual_labels_asl[i]}", fontsize=10)
    plt.axis('off')
    
    # Display predicted image
    plt.subplot(2, 10, i + 11)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_labels_asl[i]}", fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.show()


# In[3]:


# Ensure the necessary packages are installed
get_ipython().system('pip install pydot graphviz')

from tensorflow.keras.utils import plot_model
import tensorflow as tf

model = tf.keras.models.load_model("americanSignLanguage.keras")

# Generate and save the model architecture diagram
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)


# In[4]:


from graphviz import Digraph

# Create a directed graph
dot = Digraph(comment='Sign Language Model Flowchart')

# Add nodes for each stage in your pipeline
dot.node('A', 'Load Data (CSV files)')
dot.node('B', 'Preprocess Data (Normalization, Reshape)')
dot.node('C', 'Build Model (CNN Architecture)')
dot.node('D', 'Compile Model (Optimizer, Loss, Metrics)')
dot.node('E', 'Augment Data (ImageDataGenerator)')
dot.node('F', 'Train Model (Fit Model with Augmented Data)')
dot.node('G', 'Evaluate Model (Testing Data)')
dot.node('H', 'Make Predictions (On Test Set)')
dot.node('I', 'Visualize Results (Predictions vs Actual)')

# Add edges to represent the flow of the process
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G')
dot.edge('G', 'H')
dot.edge('H', 'I')

# Render the flowchart as a PNG file
dot.render('model_flowchart', format='png', cleanup=True)

# Display the flowchart image
from PIL import Image
Image.open('model_flowchart.png')


# In[5]:


get_ipython().system('pip freeze > requirements.txt')


# In[ ]:




