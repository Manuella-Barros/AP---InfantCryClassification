from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
import sklearn.preprocessing as sk_preprocessing

def mlp(X_train, y_train, X_val, y_val):
    print("Iniciando o treinamento do modelo MLP...")

    ## PADRONIZA O TAMANHO OS DADOS DE ENTRADA =================================
    x_train_norm, y_train_norm, x_val_norm, y_val_norm = normalize_data(X_train, y_train, X_val, y_val)

    ## CRIA O MODELO SEQUENCIAL, que é uma pilha linear de camadas. ============
    model = Sequential([
        Dense(128, activation='relu', input_shape=(302, 39)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        GlobalAveragePooling1D(),
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
        callbacks=[early_stop]
    )
    print("history:", history.history)

    ## SALVA O MODELO =========================================================
    # model.save("mlp_model.keras")

    ## CARREGA O MODELO =======================================================
    # model = load_model("mlp_model.keras")

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