from graphviz import Digraph

def generate_nn_graph(num_layers, neurons_per_layer, num_inputs):
    """
    Generates a visually correct graph representation of a neural network using the Graphviz library,
    addressing specific aesthetic preferences and structural adjustments.
    
    Args:
    - num_layers (int): The number of layers in the neural network, including the output layer but not the input.
    - neurons_per_layer (list of int): A list where each element represents the number of neurons in the corresponding layer.
    - num_inputs (int): The number of input features.
    
    Returns:
    - Digraph object: A directed graph that visually represents the neural network with specified adjustments.
    """
    
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    dot.attr('node', shape='circle', style='filled', fontname='Helvetica')
    dot.node('input', label=f'Input Layer\n{num_inputs}', shape='rect', style='filled', fillcolor='lightgrey', fontname='Helvetica')

    for layer in range(num_layers):
        with dot.subgraph(name=f'cluster_{layer}') as c:
            if layer == num_layers - 1:
                c.attr(color='lightcoral', style='filled', fillcolor='lightcoral', label=f'Output Layer')
            else:
                c.attr(color='lightblue', style='filled', fillcolor='lightblue', label=f'Hidden Layer {layer+1}')
            
            for neuron in range(neurons_per_layer[layer]):
                c.node(f'layer_{layer}_neuron_{neuron}', f'N{neuron + 1}')

    dot.edge('input', f'layer_0_neuron_0', style='solid')

    for layer in range(1, num_layers):
        dot.edge(f'layer_{layer - 1}_neuron_0', f'layer_{layer}_neuron_0', style='solid')

    return dot
