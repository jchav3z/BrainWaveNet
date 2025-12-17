import matplotlib.pyplot as plt
import numpy as np
import os

def generate_performance_chart():
    """
    Genera un gráfico de barras que muestra el F1-score simulado
    por cada una de las 5 etapas del sueño.
    """
    classes = ['WAKE', 'N1', 'N2', 'N3', 'REM']
    f1_scores = [0.95, 0.78, 0.90, 0.85, 0.84]
    
    f1_macro = np.mean(f1_scores)

    # Crear el gráfico
    plt.figure(figsize=(9, 6))
    bars = plt.bar(classes, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

    plt.axhline(f1_macro, color='r', linestyle='--', linewidth=1, label=f'F1-score Macro ({f1_macro:.3f})')
    
    plt.title(f'Rendimiento F1-score por Clase (Simulado)\nF1-score Macro Total: {f1_macro:.3f}', fontsize=14)
    plt.ylabel('F1-score')
    plt.xlabel('Etapa del Sueño')
    plt.ylim(0.0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower left')

    filename = 'f1_score_per_class.png'
    plt.savefig(filename)
    print(f"\n--- ÉXITO ---\nGráfico de rendimiento generado y guardado como '{filename}'.")


if __name__ == '__main__':
    generate_performance_chart()