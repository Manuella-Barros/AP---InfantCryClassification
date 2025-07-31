from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def generate__confusion_matrix(y_true_classes, y_pred_classes, class_names):
    """
    Gera uma matriz de confusão a partir dos rótulos verdadeiros e previstos.

    Args:
        y_true_classes (list): Lista de rótulos verdadeiros.
        y_pred_classes (list): Lista de rótulos previstos.
        labels (list, optional): Lista de rótulos únicos. Se None, será extraída dos dados.

    Returns:
        np.ndarray: Matriz de confusão.
    """

    print("Matriz de Confusão:")
    print(f"rotulos reais de validação: {y_true_classes}")
    print(f"rotulos previstos: {y_pred_classes}")

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.title("Matriz de Confusão")