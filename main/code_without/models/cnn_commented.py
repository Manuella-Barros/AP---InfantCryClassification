from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
import sklearn.preprocessing as sk_preprocessing

## Versao com os comentários de explicação do código

def cnn(X_train, y_train, X_val, y_val):
    """
    Treina um modelo CNN (Convolutional Neural Network) com os dados fornecidos.
    
    Args:
        X_train: Dados de treinamento (features).
        y_train: Rótulos de treinamento (classes).
        X_val: Dados de validação (features).
        y_val: Rótulos de validação (classes).
    
    - Uma CNN é um tipo de rede neural usada para reconhecer padrões em dados com estrutura em grade, como imagens ou espectrogramas de áudio (como o MFCC).
    - Os MFCCs transformam os áudios em uma espécie de "imagem sonora", e a CNN é especialista em detectar padrões espaciais nessa imagem. Isso ajuda a:
        - Reconhecer variações no som ao longo do tempo
        - Aprender padrões diferentes para cada tipo de choro (dor, fome, etc)
        - Melhorar a precisão na classificação.
    """
    print("Iniciando o treinamento do modelo CNN...")

    ## PADRONIZA O TAMANHO OS DADOS DE ENTRADA =================================
    x_train_norm, y_train_norm, x_val_norm, y_val_norm = normalize_data(X_train, y_train, X_val, y_val)

    ## CRIA O MODELO SEQUENCIAL, que é uma pilha linear de camadas. ============
    model = Sequential([ # Cria uma rede neural sequencial, camada por camada.
        # input_shape deve ser (tempo: frames = 302, n_mfcc = 39, numero de canais = 1) para CNNs. Para escala cinza é 1, para RGB é 3.
        # Conv2D é uma camada de convolução que aplica filtros para extrair características e padrões dos dados de entrada.
        #   - (3, 3) é o tamanho do filtro, e 'relu' é a função de ativação.
        Conv2D(32, (3, 3), activation='relu', input_shape=(302, 39, 1)),
        MaxPooling2D((2, 2)), # MaxPooling2D Reduz a imagem (resolução), pegando apenas os valores mais altos de cada região. Isso diminui o custo de processamento e foca nos padrões mais importantes.
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        GlobalAveragePooling1D(), # os passos anteriores a predição é feita para cada frame, mas no final queremos uma única predição por áudio, então usamos o GlobalAveragePooling1D para reduzir a dimensão dos dados. Tira a média das features, resultando menos dados e menos parâmetros na saida.
        Dense(64, activation='relu'), # Faz a classificação baseada no que foi aprendido
        Dropout(0.5), # Dropout é uma técnica de regularização que ajuda a evitar o overfitting, desligando aleatoriamente uma fração dos neurônios durante o treinamento. Aqui desligou 50% dos neurônios.
        Dense(5, activation='softmax') # probabilidade entre as 5 classes de saída
    ])

    # Summary mostra um resumo da arquitetura do modelo, incluindo o número de parâmetros treináveis, não treináveis e saida de cada camada.
    print("Resumo do modelo:")
    model.summary()

    ## COMPILA O MODELO =======================================================
    # Compila o modelo, especificando: 
    # Otimizador -> Adam (Algoritmo que ajusta os pesos da rede neural para minimizar o erro),
    # Função de perda -> sparse_categorical_crossentropy,
    # Métricas a serem monitoradas durante o treinamento -> 'accuracy' (precisão).
    print("Compilando o modelo...")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    ## EARLY STOPPING ==========================================================
    # EarlyStopping é uma técnica para evitar o overfitting (quando o modelo aprende muito bem os dados de treinamento, mas não generaliza bem para novos dados).
    # É um recurso do Keras que para automaticamente o treinamento do modelo quando a métrica que você escolheu para de melhorar depois de um certo número de épocas.
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=3, # número de épocas sem melhoria antes de parar o treinamento,
        restore_best_weights=True
    )

    ## TREINA O MODELO =========================================================
    print("Treinando o modelo...")
    history = model.fit(
        x_train_norm, 
        y_train_norm, 
        validation_data=(x_val_norm, y_val_norm), 
        epochs=10, # Número de vezes que o modelo passa por todo o conjunto de dados de treinamento.
        batch_size= 32, # Tamanho do lote, ou seja, o número de amostras que o modelo processa antes de atualizar os pesos.
        callbacks=[early_stop]
    )
    print("history:", history.history)

    ## SALVA O MODELO =========================================================
    ## Para salvar o modelo treinado
    # model.save("cnn_model.keras")

    ## CARREGA O MODELO =======================================================
    ## Para carregar o modelo treinado
    # model = load_model("cnn_model.keras")

def normalize_data(X_train, y_train, X_val, y_val):
    ## PADRONIZA O TAMANHO DOS VETORES DE ENTRADA ==============================
    # Isso garante que todos os vetores de entrada tenham o mesmo tamanho, o que é necessário para treinar o modelo. (302,39)
    max_len = 302

    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', dtype='float32')
    X_val = pad_sequences(X_val, maxlen=max_len, padding='post', dtype='float32')

    ## PADRONIZA AS CLASSES ====================================================
    ## Converte as classes de string para números inteiros
    encoder = sk_preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)

    return X_train, y_train, X_val, y_val