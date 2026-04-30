import os
import json
import heatmap
import numpy as np
from scipy.stats import spearmanr

def calculate_model_diffs():
    base_model_transformer_heatmaps = os.listdir("results/base-model/transformer-heatmaps")
    ft_model_transformer_heatmaps = os.listdir("results/ft-model/transformer-heatmaps")

    mutual_heatmaps = set(base_model_transformer_heatmaps).intersection(set(ft_model_transformer_heatmaps))

    os.makedirs("results/model-diffs/transformer-heatmaps", exist_ok=True)
    os.makedirs("results/model-diffs/correlation-tables", exist_ok=True)

    for mutual_heatmap in mutual_heatmaps:
        with open(f"results/base-model/transformer-heatmaps/{mutual_heatmap}/normalized.json", "r") as base_model_heatmap_file:
            with open(f"results/ft-model/transformer-heatmaps/{mutual_heatmap}/normalized.json", "r") as ft_model_heatmap_file:
                base_model_heatmap_data = json.load(base_model_heatmap_file)
                ft_model_heatmap_data = json.load(ft_model_heatmap_file)

                model_diff_heatmap_data = [
                    [
                        ft_model_heatmap_data[layer_index][head_index] - base_model_heatmap_data[layer_index][head_index]
                        for head_index in range(len(base_model_heatmap_data[0]))
                    ]
                    for layer_index in range(len(base_model_heatmap_data))
                ]

                heatmap.transformer_heatmap(
                    f"results/base-model/transformer-heatmaps/{mutual_heatmap}.png",
                    f"{mutual_heatmap}\n(Normalized Feature Attention in Base Model)",
                    base_model_heatmap_data
                )

                heatmap.transformer_heatmap(
                    f"results/ft-model/transformer-heatmaps/{mutual_heatmap}.png",
                    f"{mutual_heatmap}\n(Normalized Feature Attention in Fine-Tuned Model)",
                    ft_model_heatmap_data
                )

                heatmap.transformer_heatmap(
                    f"results/model-diffs/transformer-heatmaps/{mutual_heatmap}.png",
                    f"{mutual_heatmap}\n(Difference Between Normalized Feature Attention in Fine Tuned vs. Base Model)",
                    model_diff_heatmap_data
                )

                flat_base_model_heatmap_data = np.array(base_model_heatmap_data).flatten()
                flat_ft_model_heatmap_data = np.array(ft_model_heatmap_data).flatten()
                flat_model_diff_heatmap_data = np.array(model_diff_heatmap_data).flatten()

                sorted_base_model_indices = np.argsort(flat_base_model_heatmap_data)[::-1]

                correlation_table = {
                    "ft": {},
                    "diff": {}
                }

                k_values = [ 5, 10, 25, 50 ]

                for k in k_values:
                    top_k_base_model_indices = sorted_base_model_indices[:k]

                    correlation_table["ft"][k] = spearmanr(flat_base_model_heatmap_data[top_k_base_model_indices], flat_ft_model_heatmap_data[top_k_base_model_indices]).statistic
                    correlation_table["diff"][k] = spearmanr(flat_base_model_heatmap_data[top_k_base_model_indices], flat_model_diff_heatmap_data[top_k_base_model_indices]).statistic

                correlation_table["ft"]["all"] = spearmanr(flat_base_model_heatmap_data, flat_ft_model_heatmap_data).statistic
                correlation_table["diff"]["all"] = spearmanr(flat_base_model_heatmap_data, flat_model_diff_heatmap_data).statistic

                with open(f"results/model-diffs/correlation-tables/{mutual_heatmap}.json", "w") as correlation_table_file:
                    json.dump(correlation_table, correlation_table_file, indent=4)


if __name__ == "__main__": calculate_model_diffs()