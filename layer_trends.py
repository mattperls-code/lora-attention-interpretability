import json
import os
import matplotlib.pyplot as plt

def from_heatmap_file(path: str):
    with open(path, "r") as heatmap_file:
        heatmap_data = json.load(heatmap_file)

        return [
            sum(layer) / len(layer)
            for layer in heatmap_data
        ]
    
def apply_windowing(layer_data: list[float], window_size: int):
    window_results = []

    for layer_index in range(len(layer_data) - window_size + 1):
        window_results.append(0)

        for window_offset in range(window_size):
            window_results[-1] += layer_data[layer_index + window_offset]

        window_results[-1] /= window_size

    return window_results

def plot_layer_data(title: str, ylabel: str, data: list[float], output_path: str, layer_window_size: int = 1):
    plt.clf()

    plt.figure(figsize=(10, 8))
    plt.title(title, pad=20)
    plt.xlabel("Layer Index", labelpad=20)
    plt.ylabel(ylabel, labelpad=20)
    plt.axhline(y=0, color='black', linestyle='--')
    plt.grid(True, axis='both', color='lightgray')

    plt.plot(range(len(data)), data)

    if layer_window_size > 1:
        plt.xticks(
            ticks=range(len(data)),
            labels=[f"{i + 1}-{i + layer_window_size}" for i in range(len(data))],
            rotation=45,
            ha="right"
        )

    else:
        plt.xticks(
            ticks=range(len(data)),
            labels=range(1, len(data) + 1),
            rotation=45,
            ha="right"
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    model_data_paths = [ "base-model", "ft-model" ]

    for model_data_path in model_data_paths:
        for feature_name in os.listdir(f"results/{model_data_path}/transformer-heatmaps"):
            if os.path.isdir(f"results/{model_data_path}/transformer-heatmaps/{feature_name}"):
                os.makedirs(f"results/{model_data_path}/layers/{feature_name}", exist_ok=True)

                feature_data_path = f"results/{model_data_path}/transformer-heatmaps/{feature_name}/normalized.json"

                feature_layer_data = from_heatmap_file(feature_data_path)

                with open(f"results/{model_data_path}/layers/{feature_name}/layer.json", "w") as layer_data_file:
                    json.dump(feature_layer_data, layer_data_file)

                plot_layer_data(
                    f"{feature_name}\n(Average Normalized Feature Attention Per Layer)",
                    "Average Normalized Feature Attention Per Layer",
                    feature_layer_data,
                    f"results/{model_data_path}/layers/{feature_name}/layer.png"
                )

                layer_window_sizes = [ 2, 3, 4, 6 ]

                for layer_window_size in layer_window_sizes:
                    feature_window_data = apply_windowing(feature_layer_data, layer_window_size)

                    with open(f"results/{model_data_path}/layers/{feature_name}/window-size{layer_window_size}.json", "w") as window_data_file:
                        json.dump(feature_window_data, window_data_file)

                    plot_layer_data(
                        f"{feature_name}\n(Average Normalized Feature Attention Per {layer_window_size} Layer Window)",
                        "Average Normalized Feature Attention Per Window",
                        feature_window_data,
                        f"results/{model_data_path}/layers/{feature_name}/window-size{layer_window_size}.png",
                        layer_window_size=layer_window_size
                    )

    data_file_paths = [
        ("layer", "Layer", 1),
        ("window-size2", "Window", 2),
        ("window-size3", "Window", 3),
        ("window-size4", "Window", 4),
        ("window-size6", "Window", 6)
    ]

    for feature_name in os.listdir(f"results/base-model/layers"):
        if os.path.isdir(f"results/ft-model/layers/{feature_name}"):
            os.makedirs(f"results/model-diffs/layers/{feature_name}", exist_ok=True)

            for (data_file_path, grouping_name, layer_window_size) in data_file_paths:
                with open(f"results/base-model/layers/{feature_name}/{data_file_path}.json", "r") as base_model_data_file:
                    with open(f"results/ft-model/layers/{feature_name}/{data_file_path}.json", "r") as ft_model_data_file:
                        base_model_data = json.load(base_model_data_file)
                        ft_model_data = json.load(ft_model_data_file)

                        model_diffs_data = [
                            ft_model_data[layer_index] - base_model_data[layer_index]
                            for layer_index in range(len(base_model_data))
                        ]

                        with open(f"results/model-diffs/layers/{feature_name}/{data_file_path}.json", "w") as diff_data_file:
                            json.dump(model_diffs_data, diff_data_file)

                        plot_layer_data(
                            f"{feature_name}\n(Difference Between Average Normalized Feature Attention in Fine Tuned vs Base Model)",
                            f"Normalized Feature Attention Difference Per {grouping_name}",
                            model_diffs_data,
                            f"results/model-diffs/layers/{feature_name}/{data_file_path}.png",
                            layer_window_size=layer_window_size
                        )