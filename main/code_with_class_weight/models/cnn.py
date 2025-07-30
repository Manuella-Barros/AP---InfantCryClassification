from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
import sklearn.preprocessing as sk_preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight

def cnn(X_train, y_train, X_val, y_val):
    print("Iniciando o treinamento do modelo CNN...")

    ## PADRONIZA O TAMANHO OS DADOS DE ENTRADA =================================
    x_train_norm, y_train_norm, x_val_norm, y_val_norm = normalize_data(X_train, y_train, X_val, y_val)

    # Calcular pesos das classes
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print("Pesos das classes:", class_weights)

    ## CRIA O MODELO SEQUENCIAL, que é uma pilha linear de camadas. ============
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(302, 39, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(5, activation='softmax') 
    ])

    print("Resumo do modelo:")
    model.summary()

    ## COMPILA O MODELO =======================================================
    print("Compilando o modelo...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ## EARLY STOPPING ==========================================================
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )

    ## TREINA O MODELO =========================================================
    print("Treinando o modelo...")
    history = model.fit(
        x_train_norm, 
        y_train_norm, 
        validation_data=(x_val_norm, y_val_norm), 
        epochs=10,
        batch_size= 32,
        callbacks=[early_stop],
        class_weight=class_weights
    )
    print("history:", history.history)

    ## AVALIA O MODELO =========================================================
    print("Avaliação do modelo:")
    y_pred = model.predict(x_val_norm).argmax(axis=1)

    # retorna as classes reiais e as classes previstas
    return y_val_norm, y_pred

    ## SALVA O MODELO =========================================================
    # model.save("cnn_model.keras")

    ## CARREGA O MODELO =======================================================
    # model = load_model("cnn_model.keras")

def normalize_data(X_train, y_train, X_val, y_val):
    ## PADRONIZA O TAMANHO DOS VETORES DE ENTRADA ==============================
    max_len = 302

    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', dtype='float32')
    X_val = pad_sequences(X_val, maxlen=max_len, padding='post', dtype='float32')

    ## PADRONIZA AS CLASSES ====================================================
    encoder = sk_preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)

    return X_train, y_train, X_val, y_val