from graphviz import Digraph

def generate_nn_graph(num_layers, neurons_per_layer, num_inputs, minimalistic=True):
    """
    Generates a visually correct graph representation of a neural network using the Graphviz library,
    addressing specific aesthetic preferences and structural adjustments. If minimalistic is True,
    only the first and last neuron of each layer are shown, with ellipsis (...) in between.
    
    Args:
    - num_layers (int): The number of layers in the neural network, including the output layer but not the input.
    - neurons_per_layer (list of int): A list where each element represents the number of neurons in the corresponding layer.
    - num_inputs (int): The number of input features.
    - minimalistic (bool): Whether to use a minimalistic representation.
    
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
            
            if minimalistic and neurons_per_layer[layer] > 2:
                c.node(f'layer_{layer}_neuron_0', f'N1')
                c.node(f'layer_{layer}_neuron_...', '...')
                c.node(f'layer_{layer}_neuron_{neurons_per_layer[layer]-1}', f'N{neurons_per_layer[layer]}')
            else:
                for neuron in range(neurons_per_layer[layer] - 1, -1, -1):
                    c.node(f'layer_{layer}_neuron_{neuron}', f'N{neuron + 1}')

    dot.edge('input', f'layer_0_neuron_0', style='solid')

    # Connecting layers
    for layer in range(1, num_layers):
        prev_layer_size = neurons_per_layer[layer - 1]
        current_layer_size = neurons_per_layer[layer]
        dot.edge(f'layer_{layer - 1}_neuron_0', f'layer_{layer}_neuron_0', style='solid')

    return dot
