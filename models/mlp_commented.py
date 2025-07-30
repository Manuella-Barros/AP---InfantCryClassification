from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
import sklearn.preprocessing as sk_preprocessing
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

## Versao com os comentários de explicação do código

def mlp(X_train, y_train, X_val, y_val):
    """
    Treina um modelo MLP (Multilayer Perceptron) com os dados fornecidos.
    
    Args:
        X_train: Dados de treinamento (features).
        y_train: Rótulos de treinamento (classes).
        X_val: Dados de validação (features).
        y_val: Rótulos de validação (classes).
    
    - MLP (Multilayer Perceptron) é  um modelo simples, rápido de treinar
    - A softamx serve para a predição de classes, onde a saída é uma probabilidade entre as classes
    - A função de ativação ReLU (Rectified Linear Unit) é uma escolha comum para camadas ocultas, pois ajuda a evitar o problema do gradiente desaparecendo e permite que o modelo aprenda relações não lineares complexas.
    """
    print("Iniciando o treinamento do modelo MLP...")

    ## PADRONIZA O TAMANHO OS DADOS DE ENTRADA =================================
    x_train_norm, y_train_norm, x_val_norm, y_val_norm = normalize_data(X_train, y_train, X_val, y_val)

    ## CALCULA OS PESOS DAS CLASSES ===========================================
    # A função compute_class_weight calcula pesos para cada classe em um problema de classificação desequilibrado. Ela calcula um peso proporcionalmente inverso à frequência da classe. Quanto menos amostras a classe tem, maior será seu peso.

    class_weights = compute_class_weight(
        class_weight='balanced', # 'balanced' significa que os pesos serão calculados de forma a equilibrar as classes
        classes=np.unique(y_train_norm), # garante que os pesos sejam calculados exatamente para as classes que estão presentes no treino.
        y=y_train_norm #vetor com os rótulos das amostras de treino
    )
    class_weights_dict = dict(enumerate(class_weights)) # {0: 0.5, 1: 1.0, 2: 2.0, 3: 0.7, 4: 1.5}
    print("Pesos das classes:", class_weights_dict)

    ## CRIA O MODELO SEQUENCIAL, que é uma pilha linear de camadas. ============
    model = Sequential([ # Cria uma rede neural sequencial, camada por camada.
        Dense(128, activation='relu', input_shape=(302, 39)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        GlobalAveragePooling1D(), # nos passos anteriores a predição é feita para cada frame, mas no final queremos uma única predição por áudio, então usamos o GlobalAveragePooling1D para reduzir a dimensão dos dados.
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
        callbacks=[early_stop],
        class_weight=class_weights_dict # Pesos das classes para lidar com o desbalanceamento de classes
    )
    print("history:", history.history)

    ## SALVA O MODELO =========================================================
    ## Para salvar o modelo treinado
    # model.save("mlp_model.keras")

    ## CARREGA O MODELO =======================================================
    ## Para carregar o modelo treinado
    # model = load_model("mlp_model.keras")

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