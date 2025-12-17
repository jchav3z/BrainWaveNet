import graphviz
import os

def create_brainwavenet_diagram_final():
    """
    Genera un diagrama de arquitectura CNN 1D BrainWaveNet profesional,
    utilizando Graphviz, con etiquetas en español y detalles de las capas.
    """
    
    # Configuración del Grafico
    dot = graphviz.Digraph(
        comment='BrainWaveNet CNN 1D Architecture', 
        graph_attr={'rankdir': 'LR', 'splines': 'spline', 'bgcolor': 'white'},
        node_attr={'shape': 'box', 'style': 'filled', 'fontname': 'Helvetica'},
        edge_attr={'fontname': 'Helvetica'}
    )

    
    dot.node('Input', 'SEÑAL EEG\n(1, 3000 puntos)', fillcolor='#D9ED92', color='#1A5319', fontcolor='#1A5319')

    dot.node('Conv1', 
             'BLOQUE CONV. 1D\nFiltros=32\nKernel=64, Stride=16\n[BatchNorm + ReLU]\nMaxPool(4)', 
             fillcolor='#A2D2FF', color='#0077B6')

    dot.node('Conv2', 
             'BLOQUE CONV. 1D\nFiltros=64\nKernel=8, Stride=1\n[BatchNorm + ReLU]\nMaxPool(2)', 
             fillcolor='#A2D2FF', color='#0077B6')

    dot.node('Flatten', 'CAPA DE APLANAMIENTO\n(Vector 1D de Features)', fillcolor='#FEE440', color='#FF6300')

    dot.node('Dense', 
             'CAPAS DENSAS (FC)\nLineal(128) -> ReLU\nDROPOUT (0.5)\nLineal(5)', 
             fillcolor='#FFC8DD', color='#C71585')

    dot.node('Output', 
             'CLASIFICACIÓN FINAL\nWAKE | N1 | N2 | N3 | REM', 
             shape='cylinder', fillcolor='#E9FFCD', color='#3C732F', fontcolor='#3C732F')

    dot.edge('Input', 'Conv1', label='3000 puntos')
    dot.edge('Conv1', 'Conv2', label='Reducción de Dimensión')
    dot.edge('Conv2', 'Flatten', label='Features Finales')
    dot.edge('Flatten', 'Dense')
    dot.edge('Dense', 'Output', label='Probabilidades (Softmax)')

    output_filename = 'brainwavenet_diagram_final'
    try:
        dot.render(output_filename, view=False, format='png')
        print(f"\n--- ÉXITO ---\nDiagrama generado y guardado como '{output_filename}.png' en el directorio actual.")
    except graphviz.backend.execute.ExecutableNotFound:
        print("\n--- ERROR CRÍTICO ---\nEl ejecutable 'dot' de Graphviz no fue encontrado.")
        print("Asegúrate de instalar Graphviz y agregar su ruta al PATH del sistema, luego reintenta.")


if __name__ == '__main__':
    # Verificar si el entorno virtual está activo
    if 'VIRTUAL_ENV' in os.environ:
        create_brainwavenet_diagram_final()
    else:
        print("ERROR: Por favor, activa el entorno virtual (venv) antes de ejecutar el script.")