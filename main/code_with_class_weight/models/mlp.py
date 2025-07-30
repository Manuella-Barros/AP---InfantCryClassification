from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import numpy as np

def mlp(X_train, y_train, X_val, y_val):
    print("Iniciando o treinamento do modelo MLP...")

    ## PADRONIZA O TAMANHO OS DADOS DE ENTRADA =================================
    x_train_norm, y_train_norm, x_val_norm, y_val_norm = normalize_data(X_train, y_train, X_val, y_val)

    ## CRIA O MODELO SEQUENCIAL, que é uma pilha linear de camadas. ============
    model = Sequential([
        Dense(128, activation='relu', input_shape=(302, 39)),
        Dropout(0.3),  # <--- Adiciona dropout
        Dense(64, activation='relu'),
        Dropout(0.2),  # <--- Adiciona dropout
        Dense(32, activation='relu'),
        Flatten(),
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
        patience=2,  # número de épocas sem melhoria antes de parar o treinamento
        restore_best_weights=True
    )

    # Calcular pesos das classes
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))
    print("Pesos das classes:", class_weights)

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

    return y_val_norm, y_pred

    ## SALVA O MODELO =========================================================
    # model.save("mlp_model.keras")

    ## CARREGA O MODELO =======================================================
    # model = load_model("mlp_model.keras")

def normalize_data(X_train, y_train, X_val, y_val):
    ## PADRONIZA O TAMANHO DOS VETORES DE ENTRADA
    max_len = 302
    
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', dtype='float32')
    X_val = pad_sequences(X_val, maxlen=max_len, padding='post', dtype='float32')

    ## PADRONIZA AS CLASSES - CORREÇÃO AQUI
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)  # Converte textos para números (0, 1, 2...)
    y_val = encoder.transform(y_val)
    
    return X_train, y_train, X_val, y_val