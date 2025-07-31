import matplotlib.pyplot as plt

def plot_history(history):
    """
    Plota o histórico de treinamento com acurácia e perda para treino e validação.
    
    Args:
        history: objeto retornado pelo model.fit() (history.history)
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, acc, label='train_accuracy', color='skyblue')
    plt.plot(epochs, val_acc, label='val_accuracy', color='orange')
    plt.plot(epochs, loss, label='train_loss', color='green')
    plt.plot(epochs, val_loss, label='val_loss', color='red')

    plt.title('Historico de Treinamento')
    plt.xlabel('Epocas')
    plt.ylabel('Valor')
    plt.legend()
