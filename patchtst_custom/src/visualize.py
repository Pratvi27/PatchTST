"""Plotting helpers — all save to disk; no interactive `plt.show()` (Linux-headless safe)."""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_style("whitegrid")


def _save_and_close(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.close(fig)


def plot_training_history(history, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
    ax.plot(epochs, history['valid_loss'], 'r-s', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    textstr = f'Final Train Loss: {history["train_loss"][-1]:.4f}\n'
    textstr += f'Final Valid Loss: {history["valid_loss"][-1]:.4f}'
    if history.get('test_loss') is not None:
        textstr += f'\nTest Loss: {history["test_loss"]:.4f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_predictions(predictions, targets, num_samples=3, num_variables=5, save_path=None):
    num_samples = min(num_samples, predictions.shape[0])
    num_variables = min(num_variables, predictions.shape[1])

    fig, axes = plt.subplots(num_samples, num_variables,
                             figsize=(4 * num_variables, 3 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_variables == 1:
        axes = axes.reshape(-1, 1)

    for i in range(num_samples):
        for j in range(num_variables):
            ax = axes[i, j]
            pred = predictions[i, j, :]
            target = targets[i, j, :]
            timesteps = range(len(pred))

            ax.plot(timesteps, target, 'b-o', label='Actual', linewidth=2, markersize=4)
            ax.plot(timesteps, pred, 'r--s', label='Predicted', linewidth=2, markersize=4)

            ax.set_xlabel('Forecast Timestep', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.set_title(f'Sample {i+1}, Variable {j+1}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            mse = np.mean((pred - target) ** 2)
            ax.text(0.05, 0.95, f'MSE: {mse:.4f}', transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_error_distribution(predictions, targets, save_path=None):
    errors = (predictions - targets).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel('Prediction Error', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    from scipy import stats
    stats.probplot(errors, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    textstr = f'Mean Error: {np.mean(errors):.4f}\n'
    textstr += f'Std Error: {np.std(errors):.4f}\n'
    textstr += f'MAE: {np.mean(np.abs(errors)):.4f}\n'
    textstr += f'RMSE: {np.sqrt(np.mean(errors**2)):.4f}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[0].text(0.05, 0.95, textstr, transform=axes[0].transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_variable_performance(predictions, targets, variable_names=None, save_path=None):
    num_variables = predictions.shape[1]

    mse_per_variable = []
    for i in range(num_variables):
        mse = np.mean((predictions[:, i, :] - targets[:, i, :]) ** 2)
        mse_per_variable.append(mse)

    if variable_names is None:
        variable_names = [f'Var {i+1}' for i in range(num_variables)]

    if num_variables > 20:
        top_indices = np.argsort(mse_per_variable)[-20:]
        mse_per_variable = [mse_per_variable[i] for i in top_indices]
        variable_names = [variable_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(12, max(6, len(mse_per_variable) * 0.3)))

    colors = plt.cm.viridis(np.linspace(0, 1, len(mse_per_variable)))
    bars = ax.barh(range(len(mse_per_variable)), mse_per_variable, color=colors)

    ax.set_yticks(range(len(mse_per_variable)))
    ax.set_yticklabels(variable_names, fontsize=9)
    ax.set_xlabel('MSE', fontsize=12)
    ax.set_title('MSE per Variable (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    for i, (bar, value) in enumerate(zip(bars, mse_per_variable)):
        ax.text(value, i, f' {value:.4f}', va='center', fontsize=8)

    fig.tight_layout()
    _save_and_close(fig, save_path)


def plot_forecast_comparison(predictions, targets, sample_idx=0,
                             variable_indices=None, save_path=None):
    if variable_indices is None:
        variable_indices = list(range(min(6, predictions.shape[1])))

    num_vars = len(variable_indices)
    cols = 2
    rows = (num_vars + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if num_vars > 1 else [axes]

    for idx, var_idx in enumerate(variable_indices):
        ax = axes[idx]

        pred = predictions[sample_idx, var_idx, :]
        target = targets[sample_idx, var_idx, :]
        timesteps = range(len(pred))

        ax.plot(timesteps, target, 'b-o', label='Ground Truth',
                linewidth=2.5, markersize=6, alpha=0.7)
        ax.plot(timesteps, pred, 'r--s', label='Prediction',
                linewidth=2.5, markersize=6, alpha=0.7)
        ax.fill_between(timesteps, target, pred, alpha=0.2, color='gray')

        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))

        ax.set_xlabel('Forecast Timestep', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'Variable {var_idx} | MSE: {mse:.4f} | MAE: {mae:.4f}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    for idx in range(num_vars, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'Forecast Comparison - Sample {sample_idx}',
                 fontsize=16, fontweight='bold', y=1.00)
    fig.tight_layout()
    _save_and_close(fig, save_path)
