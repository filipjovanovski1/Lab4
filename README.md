# Lab4

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

data = pd.read_csv('pollution_dataset.csv')
data.head()

enc = LabelEncoder()
data['Air Quality'] = enc.fit_transform(data['Air Quality'])

data.head()

data.isnull().sum()

X = data.drop('Air Quality', axis=1)
y = data['Air Quality']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

def build_model(layers, activations, input_dim):
    model = Sequential()
    for i, (neurons, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(Dense(neurons, activation=activation, input_dim=input_dim))
        else:
            model.add(Dense(neurons,activation=activation))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model_1 = build_model(layers=[32, 16], activations=['relu', 'relu'], input_dim=X_train.shape[1])
history_1 = model_1.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2)

model_2 = build_model(layers=[64, 32, 16], activations=['tanh', 'relu', 'relu'], input_dim=X_train.shape[1])
history_2 = model_2.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2)

model_3 = build_model(layers=[128, 64, 32, 16], activations=['relu', 'relu', 'relu', 'relu'], input_dim=X_train.shape[1])
history_3 = model_3.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.2)

print("Model 1 Performance")
model_1_eval = model_1.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {model_1_eval[1]:.2f}")

print("Model 2 Performance")
model_2_eval = model_2.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {model_2_eval[1]:.2f}")

print("Model 3 Performance")
model_3_eval = model_3.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy: {model_3_eval[1]:.2f}")

y_pred = np.argmax(model_3.predict(X_test), axis=1)
print("Classification Report for Model 3")
print(classification_report(y_test, y_pred, target_names=enc.classes_))

def plot_history(histories, labels):
    plt.figure(figsize=(14, 6))

    colors = ['red', 'blue', 'green'] 
    plt.subplot(1,2,1)
    for index,(history, label) in enumerate(zip(histories, labels)):
        color = colors[index % len(colors)]
        plt.plot(history.history['accuracy'], label=f'{label} - Training', color=color)
        plt.plot(history.history['val_accuracy'], label=f'{label} - Validation', linestyle='dashed', color=color)
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    for index,(history, label) in enumerate(zip(histories, labels)):
        color = colors[index % len(colors)]
        plt.plot(history.history['loss'], label=f'{label} - Training', color=color)
        plt.plot(history.history['val_loss'], label=f'{label} - Validation', linestyle='dashed', color=color)
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history([history_1, history_2, history_3], labels=["Model 1", "Model 2", "Model 3"])

