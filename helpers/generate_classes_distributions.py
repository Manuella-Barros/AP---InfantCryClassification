import matplotlib.pyplot as plt
from collections import Counter

def generate_classes_distributions(y_train, y_test):
    """
    Plota a distribuição das classes nos conjuntos de treino e teste.
    
    Args:
        y_train: Rótulos de treinamento (classes).
        y_test: Rótulos de teste (classes).
    """

    print("Distribuição de classes nos conjuntos de treino e teste:")
    print(f"Teste: {Counter(y_test)}")
    print(f"Treino: {Counter(y_train)}")

    plt.figure(figsize=(10,4))
    plt.suptitle('Distribuição de classes')
    plt.subplot(1, 2, 1)
    plt.hist(y_train, bins=5)
    plt.title('Treino')
    plt.subplot(1, 2, 2)
    plt.hist(y_test, bins=5)
    plt.title('Validação')