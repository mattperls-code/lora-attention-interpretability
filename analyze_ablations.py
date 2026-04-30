import math
import os
import json
import reranker
from heatmap import transformer_heatmap
import layer_trends

def mean_score_margin(rel_score: float, nrel_scores: list[float]):
    return -(rel_score - sum(nrel_scores) / len(nrel_scores))

def min_score_margin(rel_score: float, nrel_scores: list[float]):
    return -(rel_score - max(nrel_scores))

def categorical_cross_entropy(rel_score: float, nrel_scores: list[float]):
    rel_score_exp = math.exp(rel_score)
    nrel_scores_exp = [ math.exp(nrel_score) for nrel_score in nrel_scores ]

    return -math.log(rel_score_exp / (rel_score_exp + sum(nrel_scores_exp)))

def reciprocal_rank(rel_score: float, nrel_scores: list[float]):
    all_scores = nrel_scores + [ rel_score ]
    rank = sorted(all_scores, reverse=True).index(rel_score) + 1

    return -1.0 / rank

def ndcg(rel_score: float, nrel_scores: list[float]):
    all_scores = nrel_scores + [ rel_score ]
    
    rank = sorted(all_scores, reverse=True).index(rel_score)

    return -1.0 / math.log2(rank + 2)

evaluation_metrics = {
    "Mean Score Margin": mean_score_margin,
    "Min Score Margin": min_score_margin,
    "Categorical Cross Entropy": categorical_cross_entropy,
    "Reciprocal Rank": reciprocal_rank,
    "NDCG": ndcg
}

def analyze_ablation(ablation_data_path: str):
    with open(ablation_data_path, "r") as ablation_data_file:
        ablation_data = json.load(ablation_data_file)

        evaluation_metric_scores = {}

        for evaluation_metric, calculate_metric in evaluation_metrics.items():
            samples = []

            for logits in ablation_data:
                samples.append(calculate_metric(logits["rel"], logits["nrel"]))

            evaluation_metric_scores[evaluation_metric] = sum(samples) / len(samples)
        
        return evaluation_metric_scores
    
def normalize_head_data(head_data: list[list[float]]):
    max_abs = max(abs(value) for layer in head_data for value in layer)

    return [[ value / max_abs for value in layer ] for layer in head_data]

def analyze_omit_single_head_ablations():
    evaluation_metric_control_values = analyze_ablation("results/ablation/none.json")

    evaluation_metric_heatmaps = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }

    for layer_index in range(reranker.ft_model.config.num_hidden_layers):
        for evaluation_metric in evaluation_metrics.keys():
            evaluation_metric_heatmaps[evaluation_metric].append([])

        for head_index in range(reranker.ft_model.config.num_attention_heads):
            evaluation_metric_scores = analyze_ablation(f"results/ablation/layer{layer_index}-head{head_index}.json")

            for evaluation_metric, metric_score in evaluation_metric_scores.items():
                evaluation_metric_heatmaps[evaluation_metric][-1].append(metric_score - evaluation_metric_control_values[evaluation_metric])

    os.makedirs("results/ablation-analysis/omit-head", exist_ok=True)

    for evaluation_metric, metric_heatmap in evaluation_metric_heatmaps.items():
        transformer_heatmap(
            f"results/ablation-analysis/omit-head/{evaluation_metric}.png",
            f"Average Increase in {evaluation_metric} Loss\nWith Single Head Omit Ablation",
            normalize_head_data(metric_heatmap)
        )

# TODO: abstract the following analysis routines better

def analyze_omit_single_layer_ablations():
    evaluation_metric_control_values = analyze_ablation("results/ablation/none.json")

    evaluation_metric_progressions = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }

    for layer_index in range(reranker.ft_model.config.num_hidden_layers):
        evaluation_metric_scores = analyze_ablation(f"results/ablation/layer{layer_index}.json")

        for evaluation_metric, metric_score in evaluation_metric_scores.items():
            evaluation_metric_progressions[evaluation_metric].append(metric_score - evaluation_metric_control_values[evaluation_metric])

    os.makedirs("results/ablation-analysis/omit-layer", exist_ok=True)

    for evaluation_metric in evaluation_metrics.keys():
        with open(f"results/ablation-analysis/omit-layer/{evaluation_metric}.json", "w") as metric_progressions_file:
            json.dump(evaluation_metric_progressions[evaluation_metric], metric_progressions_file)

        layer_trends.plot_layer_data(
            f"Average Increase in {evaluation_metric} Loss\nWith Single Layer Omit Ablation",
            f"Average Increase in {evaluation_metric} Loss",
            evaluation_metric_progressions[evaluation_metric],
            f"results/ablation-analysis/omit-layer/{evaluation_metric}.png"
        )

def analyze_keep_single_layer_ablations():
    evaluation_metric_control_values = analyze_ablation("results/ablation/none.json")

    evaluation_metric_progressions = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }

    for layer_index in range(reranker.ft_model.config.num_hidden_layers):
        evaluation_metric_scores = analyze_ablation(f"results/ablation/keep-layer{layer_index}.json")

        for evaluation_metric, metric_score in evaluation_metric_scores.items():
            evaluation_metric_progressions[evaluation_metric].append(metric_score - evaluation_metric_control_values[evaluation_metric])

    os.makedirs("results/ablation-analysis/keep-layer", exist_ok=True)

    for evaluation_metric in evaluation_metrics.keys():
        with open(f"results/ablation-analysis/keep-layer/{evaluation_metric}.json", "w") as metric_progressions_file:
            json.dump(evaluation_metric_progressions[evaluation_metric], metric_progressions_file)

        layer_trends.plot_layer_data(
            f"Average Increase in {evaluation_metric} Loss\nWith Single Layer Keep Ablation",
            f"Average Increase in {evaluation_metric} Loss",
            evaluation_metric_progressions[evaluation_metric],
            f"results/ablation-analysis/keep-layer/{evaluation_metric}.png"
        )

def analyze_omit_window_ablations():
    evaluation_metric_control_values = analyze_ablation("results/ablation/none.json")

    layer_window_sizes = [ 2, 3, 4, 6 ]
    
    for layer_window_size in layer_window_sizes:
        evaluation_metric_progressions = {
            evaluation_metric: []
            for evaluation_metric in evaluation_metrics.keys()
        }

        for layer_window_index in range(reranker.ft_model.config.num_hidden_layers - layer_window_size + 1):
            evaluation_metric_scores = analyze_ablation(f"results/ablation/window{layer_window_index}-size{layer_window_size}.json")

            for evaluation_metric, metric_score in evaluation_metric_scores.items():
                evaluation_metric_progressions[evaluation_metric].append(metric_score - evaluation_metric_control_values[evaluation_metric])

        os.makedirs(f"results/ablation-analysis/omit-window/size{layer_window_size}", exist_ok=True)

        for evaluation_metric in evaluation_metrics.keys():
            with open(f"results/ablation-analysis/omit-window/size{layer_window_size}/{evaluation_metric}.json", "w") as metric_progressions_file:
                json.dump(evaluation_metric_progressions[evaluation_metric], metric_progressions_file)

            layer_trends.plot_layer_data(
                f"Average Increase in {evaluation_metric} Loss\nWith {layer_window_size} Layer Window Omit Ablation",
                f"Average Increase in {evaluation_metric} Loss",
                evaluation_metric_progressions[evaluation_metric],
                f"results/ablation-analysis/omit-window/size{layer_window_size}/{evaluation_metric}.png",
                layer_window_size=layer_window_size
            )

def analyze_keep_window_ablations():
    evaluation_metric_control_values = analyze_ablation("results/ablation/none.json")

    layer_window_sizes = [ 2, 3, 4, 6 ]
    
    for layer_window_size in layer_window_sizes:
        evaluation_metric_progressions = {
            evaluation_metric: []
            for evaluation_metric in evaluation_metrics.keys()
        }

        for layer_window_index in range(reranker.ft_model.config.num_hidden_layers - layer_window_size + 1):
            evaluation_metric_scores = analyze_ablation(f"results/ablation/keep-window{layer_window_index}-size{layer_window_size}.json")

            for evaluation_metric, metric_score in evaluation_metric_scores.items():
                evaluation_metric_progressions[evaluation_metric].append(metric_score - evaluation_metric_control_values[evaluation_metric])

        os.makedirs(f"results/ablation-analysis/keep-window/size{layer_window_size}", exist_ok=True)

        for evaluation_metric in evaluation_metrics.keys():
            with open(f"results/ablation-analysis/keep-window/size{layer_window_size}/{evaluation_metric}.json", "w") as metric_progressions_file:
                json.dump(evaluation_metric_progressions[evaluation_metric], metric_progressions_file)

            layer_trends.plot_layer_data(
                f"Average Increase in {evaluation_metric} Loss\nWith {layer_window_size} Layer Window Keep Ablation",
                f"Average Increase in {evaluation_metric} Loss",
                evaluation_metric_progressions[evaluation_metric],
                f"results/ablation-analysis/keep-window/size{layer_window_size}/{evaluation_metric}.png",
                layer_window_size=layer_window_size
            )

if __name__ == "__main__":
    # TODO: different eval set, i probably need to redo head evals (sigh)
    analyze_omit_single_head_ablations()

    analyze_omit_single_layer_ablations()
    analyze_keep_single_layer_ablations()
    analyze_omit_window_ablations()
    analyze_keep_window_ablations()