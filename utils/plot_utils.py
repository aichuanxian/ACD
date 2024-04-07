import matplotlib                                                                                                         
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def use_svg_display(): 
    plt.rcParams['svg.image_inline'] = True

def set_figsize(figsize = (3.5, 2.5)): 
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid(True)

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None, svg_save_path = None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.savefig(svg_save_path)

# def plot_attention_weights(att_weights, save_path):
#     # Get the average attention weights for each layer
#     avg_att_weights = [item.mean(dim=[0,1,2,3]).cpu().detach().numpy() for item in att_weights]

#     # Convert to 2D array
#     avg_att_array = np.array(avg_att_weights).reshape(-1, 1)
#     # Determine color scale min and max values
#     vmin = np.min(avg_att_weights)
#     vmax = np.max(avg_att_weights)
#     # Create a heatmap
#     plt.figure(figsize=(12, 2))
#     sns.heatmap(avg_att_array.T, annot=True, fmt=".2f",
#             xticklabels=[f'Layer {i}' for i in range(len(avg_att_weights))],
#             yticklabels=['Average Attention Weights'], vmin=0, vmax=1)

#     # Save the figure
#     plt.savefig(save_path)
#     plt.close()

def plot_attention_weights(att_weights, save_path):
    # Get the average attention weights for each layer and each head
    # for i, att in enumerate(att_weights):
    #     print(f"Layer {i}, min weight: {att.min()}, max weight: {att.max()}")
    avg_att_weights = [item.mean(dim=[0,2,3]).cpu().detach().numpy() for item in att_weights]
    
    # Normalize the weights for each layer into the range [0,1]
    normalized_avg_att_weights = [(item - np.min(item)) / (np.max(item) - np.min(item)) for item in avg_att_weights]

    # Use normalized weights instead of original averages
    att_matrix = np.vstack(normalized_avg_att_weights)
    # Stack arrays in sequence vertically (row wise)
    # att_matrix = np.vstack(avg_att_weights)

    # Transpose the matrix to swap the axes
    att_matrix = att_matrix.T

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(att_matrix, annot=True, fmt=".2f", vmin=0, vmax=1)
    
    # Set labels
    plt.xlabel('Layer')
    plt.ylabel('Attention Head')

    # Save the figure
    plt.savefig(save_path)
    plt.close()