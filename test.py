import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.regularizers import l2

# Caricamento dei dati
file_csv = 'dataset/SDSS_DR18.csv'
df = pd.read_csv(file_csv)

X = df[['u', 'g', 'r', 'i', 'z']]  # Features
y = df['class']  # Target

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def create_model(dropout_rate=0.0, neurons=64):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train_scaled.shape[1],), kernel_regularizer=l2(0.001)),
        Dropout(dropout_rate),
        Dense(neurons, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(dropout_rate),
        Dense(neurons // 2, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(dropout_rate),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=32, verbose=0)

param_grid = {
    'dropout_rate': [0.0, 0.3, 0.5],
    'neurons': [64, 128]
}

grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train_scaled, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

best_params = grid_result.best_params_

# Allenamento del modello con i migliori iperparametri
model = create_model(dropout_rate=best_params['dropout_rate'], neurons=best_params['neurons'])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f'Test accuracy: {test_accuracy}')

y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(y_test, y_pred_classes))

# Grafici di accuracy e loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

# Bar Plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='class')
plt.title('Objects Distribution')
plt.xlabel('Classe')
plt.ylabel('Frequenza')
plt.show()

# Pie Chart
plt.figure(figsize=(8, 8))
df['class'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=140)
plt.title('Percentage Objects Distribution')
plt.ylabel('')  # Rimuove l'etichetta dell'asse y per una pie chart

plt.show()
