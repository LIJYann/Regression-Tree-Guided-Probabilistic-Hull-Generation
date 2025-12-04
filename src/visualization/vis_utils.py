import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from matplotlib import font_manager

def draw_gradient_boundary(ax, model, input_range, goal, num_points=100):
    """Draw the gradient-based decision boundary."""
    from samplers.boundary_sampler import BoundaryTracer
    tracer = BoundaryTracer(model, input_range, goal=goal, num_points=num_points)
    boundary_points = tracer.trace_boundary()
    if boundary_points is not None and len(boundary_points) > 0:
        boundary_points = boundary_points.detach().cpu().numpy()
        ax.scatter(boundary_points[:, 0], boundary_points[:, 1], c='black', s=10, alpha=1.0, label='Gradient Boundary', zorder=5)

def visualize_regions(splitter_instance, ax=None, font=None, y_values=None, show_boundary_line=False):
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.gca()
    regions = splitter_instance.get_tree_regions()
    
    # Define colors for different safety states
    safety_colors = {
        1: '#90EE90',    # Light green for safe regions
        0: '#FFB6C1',    # Light red for unsafe regions
        -1: '#FFEC8B'    # Light yellow for unknown regions
    }
    
    for region in regions:
        bounds, value, depth, safety, prob, stop_reason = region
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        width = x_max - x_min
        height = y_max - y_min
        if width > 1e-6 and height > 1e-6:
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                facecolor=safety_colors[safety], alpha=0.6, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)
    
    if show_boundary_line:
        draw_gradient_boundary(ax, splitter_instance.model, splitter_instance.input_range, splitter_instance.goal)
    
    points = splitter_instance.boundary_points
    
    if y_values is not None:
        colors = ['#006400' if y >= 0 else '#8B0000' for y in y_values]  # Dark green and dark red for points
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=20, alpha=0.5, label='Boundary Points')
    else:
        ax.scatter(points[:, 0], points[:, 1], c='#8B0000', s=20, alpha=0.5, label='Boundary Points')
    
    ax.set_xlim(splitter_instance.input_range[0])
    ax.set_ylim(splitter_instance.input_range[1])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    return ax

def visualize_regions_separate(splitter_instance, ax=None, font=None, y_values=None, show_boundary_line=False):
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.gca()
    regions = splitter_instance.get_tree_regions()
    
    # Define colors for different safety states
    safety_colors = {
        1: '#90EE90',    # Light green for safe regions
        0: '#FFB6C1',    # Light red for unsafe regions
        -1: '#FFEC8B'    # Light yellow for unknown regions
    }
    
    for region in regions:
        bounds, value, depth, safety, prob, stop_reason = region
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        width = x_max - x_min
        height = y_max - y_min
        if width > 1e-6 and height > 1e-6:
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                facecolor=safety_colors[safety], alpha=0.6, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)
    
    if show_boundary_line:
        draw_gradient_boundary(ax, splitter_instance.model, splitter_instance.input_range, splitter_instance.goal)
    
    points = splitter_instance.boundary_points
    
    if y_values is not None:
        colors = ['#006400' if y >= 0 else '#8B0000' for y in y_values]  # Dark green and dark red for points
        ax.scatter(points[:, 0], points[:, 1], c=colors, s=20, alpha=0.5, label='Boundary Points')
    else:
        ax.scatter(points[:, 0], points[:, 1], c='#8B0000', s=20, alpha=0.5, label='Boundary Points')
    
    ax.set_xlim(splitter_instance.input_range[0])
    ax.set_ylim(splitter_instance.input_range[1])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    return ax

def visualize_regions_no_points(splitter_instance, ax=None, font=None, show_boundary_line=False):
    """Visualize regions without showing any sampling points."""
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.gca()
    regions = splitter_instance.get_tree_regions()
    
    # Define colors for different safety states
    safety_colors = {
        1: '#90EE90',    # Light green for safe regions
        0: '#FFB6C1',    # Light red for unsafe regions
        -1: '#FFEC8B'    # Light yellow for unknown regions
    }
    
    for region in regions:
        bounds, value, depth, safety, prob, stop_reason = region
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        width = x_max - x_min
        height = y_max - y_min
        if width > 1e-6 and height > 1e-6:
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                facecolor=safety_colors[safety], alpha=0.6, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)
    
    if show_boundary_line:
        draw_gradient_boundary(ax, splitter_instance.model, splitter_instance.input_range, splitter_instance.goal)
    
    ax.set_xlim(splitter_instance.input_range[0])
    ax.set_ylim(splitter_instance.input_range[1])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()
    return ax

def visualize_tree_structure(tree_instance, folder_name, criterion, font=None):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_instance, feature_names=['x1', 'x2'], filled=True, rounded=True, fontsize=10)
    save_path = os.path.join(folder_name, f'regression_tree_{criterion}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_two_regions(splitter_instance, folder_name, criterion, max_depth, font=None, y_values=None, show_boundary_line=True):
    if splitter_instance.tree is None:
        splitter_instance.train_decision_tree(max_depth=max_depth)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    visualize_regions(splitter_instance, ax1, font, y_values, show_boundary_line)
    visualize_regions_separate(splitter_instance, ax2, font, y_values, show_boundary_line)
    plt.tight_layout()
    save_path = os.path.join(folder_name, f'prob_regions_{criterion}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return splitter_instance.get_tree_regions()

