from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.callbacks import EarlyStopping

# MLP (Multilayer Perceptron) é  um modelo simples, rápido de treinar
# A softamx serve para a predição de classes, onde a saída é uma probabilidade entre as classes
# A função de ativação ReLU (Rectified Linear Unit) é uma escolha comum para camadas ocultas, pois ajuda a evitar o problema do gradiente desaparecendo e permite que o modelo aprenda relações não lineares complexas.

model = Sequential([ # Cria uma rede neural sequencial, camada por camada.
    Dense(128, activation='relu', input_shape=(302, 39)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    GlobalAveragePooling1D(), # nos passos anteriores a predição é feita para cada frame, mas no final queremos uma única predição por áudio, então usamos o GlobalAveragePooling1D para reduzir a dimensão dos dados.
    Dense(5, activation='softmax') # probabilidade entre as 5 classes de saída
])

# Summary mostra um resumo da arquitetura do modelo, incluindo o número de parâmetros treináveis, não treináveis e saida de cada camada.
model.summary()

# Compila o modelo, especificando: 
# Otimizador -> Adam (Algoritmo que ajusta os pesos da rede neural para minimizar o erro),
# Função de perda -> sparse_categorical_crossentropy,
# Métricas a serem monitoradas durante o treinamento -> 'accuracy' (precisão).
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# É um recurso do Keras que para automaticamente o treinamento do modelo quando a métrica que você escolheu para de melhorar depois de um certo número de épocas.
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

model.fit(
    X_train, 
    y_train, 
    validation_data=(X_val, y_val), 
    epochs=30, 
    batch_size=32,
    callbacks=[early_stop]
)