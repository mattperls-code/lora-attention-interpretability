import math
import os
import json
import reranker
from heatmap import transformer_heatmap
import layer_trends

# signage such that higher value means better performance

def mean_score_margin(rel_score: float, nrel_scores: list[float]):
    return rel_score - sum(nrel_scores) / len(nrel_scores)

def min_score_margin(rel_score: float, nrel_scores: list[float]):
    return rel_score - max(nrel_scores)

def categorical_cross_entropy(rel_score: float, nrel_scores: list[float]):
    rel_score_exp = math.exp(rel_score)
    nrel_scores_exp = [ math.exp(nrel_score) for nrel_score in nrel_scores ]

    return math.log(rel_score_exp / (rel_score_exp + sum(nrel_scores_exp)))

def reciprocal_rank(rel_score: float, nrel_scores: list[float]):
    all_scores = nrel_scores + [ rel_score ]
    rank = sorted(all_scores, reverse=True).index(rel_score) + 1

    return 1.0 / rank

def ndcg(rel_score: float, nrel_scores: list[float]):
    all_scores = nrel_scores + [ rel_score ]
    
    rank = sorted(all_scores, reverse=True).index(rel_score)

    return 1.0 / math.log2(rank + 2)

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
    flat = [ value for layer in head_data for value in layer ]
    mean = sum(flat) / len(flat)

    centered = [[ value - mean for value in layer ] for layer in head_data]
    max_abs = max(abs(value) for layer in centered for value in layer)

    return [[ value / max_abs for value in layer ] for layer in centered]

def analyze_head_ablations():
    base_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/base-model.json")
    ft_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/ft-model.json")

    base_vs_keep_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }
    base_vs_omit_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }
    ft_vs_keep_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }
    ft_vs_omit_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }

    for layer_index in range(reranker.ft_model.config.num_hidden_layers):
        for evaluation_metric in evaluation_metrics.keys():
            base_vs_keep_evaluation_diffs[evaluation_metric].append([])
            base_vs_omit_evaluation_diffs[evaluation_metric].append([])
            ft_vs_keep_evaluation_diffs[evaluation_metric].append([])
            ft_vs_omit_evaluation_diffs[evaluation_metric].append([])

        for head_index in range(reranker.ft_model.config.num_attention_heads):
            keep_head_evaluation_scores = analyze_ablation(f"results/ablation-data/keep-head/layer{layer_index}-head{head_index}.json")
            omit_head_evaluation_scores = analyze_ablation(f"results/ablation-data/omit-head/layer{layer_index}-head{head_index}.json")
            
            for evaluation_metric in evaluation_metrics.keys():
                base_vs_keep_evaluation_diffs[evaluation_metric][-1].append(keep_head_evaluation_scores[evaluation_metric] - base_model_evaluation_scores[evaluation_metric])
                base_vs_omit_evaluation_diffs[evaluation_metric][-1].append(omit_head_evaluation_scores[evaluation_metric] - base_model_evaluation_scores[evaluation_metric])
                ft_vs_keep_evaluation_diffs[evaluation_metric][-1].append(keep_head_evaluation_scores[evaluation_metric] - ft_model_evaluation_scores[evaluation_metric])
                ft_vs_omit_evaluation_diffs[evaluation_metric][-1].append(omit_head_evaluation_scores[evaluation_metric] - ft_model_evaluation_scores[evaluation_metric])

    os.makedirs("results/ablation-analysis/head/base-vs-keep", exist_ok=True)
    os.makedirs("results/ablation-analysis/head/base-vs-omit", exist_ok=True)
    os.makedirs("results/ablation-analysis/head/ft-vs-keep", exist_ok=True)
    os.makedirs("results/ablation-analysis/head/ft-vs-omit", exist_ok=True)

    for evaluation_metric in evaluation_metrics.keys():
        transformer_heatmap(
            f"results/ablation-analysis/head/base-vs-keep/{evaluation_metric}.png",
            f"{evaluation_metric} Improvement\nKeep-Head Ablation Model Against Base Model Performance",
            normalize_head_data(base_vs_keep_evaluation_diffs[evaluation_metric])
        )
        transformer_heatmap(
            f"results/ablation-analysis/head/base-vs-omit/{evaluation_metric}.png",
            f"{evaluation_metric} Improvement\nOmit-Head Ablation Model Against Base Model Performance",
            normalize_head_data(base_vs_omit_evaluation_diffs[evaluation_metric])
        )
        transformer_heatmap(
            f"results/ablation-analysis/head/ft-vs-keep/{evaluation_metric}.png",
            f"{evaluation_metric} Improvement\nKeep-Head Ablation Model Against Fine-Tuned Model Performance",
            normalize_head_data(ft_vs_keep_evaluation_diffs[evaluation_metric])
        )
        transformer_heatmap(
            f"results/ablation-analysis/head/ft-vs-omit/{evaluation_metric}.png",
            f"{evaluation_metric} Improvement\nOmit-Head Ablation Model Against Fine-Tuned Model Performance",
            normalize_head_data(ft_vs_omit_evaluation_diffs[evaluation_metric])
        )

def analyze_layer_ablations():
    base_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/base-model.json")
    ft_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/ft-model.json")

    base_vs_keep_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }
    base_vs_omit_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }
    ft_vs_keep_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }
    ft_vs_omit_evaluation_diffs = {
        evaluation_metric: []
        for evaluation_metric in evaluation_metrics.keys()
    }

    for layer_index in range(reranker.ft_model.config.num_hidden_layers):
        keep_layer_evaluation_scores = analyze_ablation(f"results/ablation-data/keep-layer/layer{layer_index}.json")
        omit_layer_evaluation_scores = analyze_ablation(f"results/ablation-data/omit-layer/layer{layer_index}.json")

        for evaluation_metric in evaluation_metrics.keys():
            base_vs_keep_evaluation_diffs[evaluation_metric].append(keep_layer_evaluation_scores[evaluation_metric] - base_model_evaluation_scores[evaluation_metric])
            base_vs_omit_evaluation_diffs[evaluation_metric].append(omit_layer_evaluation_scores[evaluation_metric] - base_model_evaluation_scores[evaluation_metric])
            ft_vs_keep_evaluation_diffs[evaluation_metric].append(keep_layer_evaluation_scores[evaluation_metric] - ft_model_evaluation_scores[evaluation_metric])
            ft_vs_omit_evaluation_diffs[evaluation_metric].append(omit_layer_evaluation_scores[evaluation_metric] - ft_model_evaluation_scores[evaluation_metric])

    os.makedirs("results/ablation-analysis/layer/base-vs-keep", exist_ok=True)
    os.makedirs("results/ablation-analysis/layer/base-vs-omit", exist_ok=True)
    os.makedirs("results/ablation-analysis/layer/ft-vs-keep", exist_ok=True)
    os.makedirs("results/ablation-analysis/layer/ft-vs-omit", exist_ok=True)
    
    for evaluation_metric in evaluation_metrics.keys():
        with open(f"results/ablation-analysis/layer/base-vs-keep/{evaluation_metric}.json", "w") as evaluation_trends_file:
            json.dump(base_vs_keep_evaluation_diffs[evaluation_metric], evaluation_trends_file)
        with open(f"results/ablation-analysis/layer/base-vs-omit/{evaluation_metric}.json", "w") as evaluation_trends_file:
            json.dump(base_vs_omit_evaluation_diffs[evaluation_metric], evaluation_trends_file)
        with open(f"results/ablation-analysis/layer/ft-vs-keep/{evaluation_metric}.json", "w") as evaluation_trends_file:
            json.dump(ft_vs_keep_evaluation_diffs[evaluation_metric], evaluation_trends_file)
        with open(f"results/ablation-analysis/layer/ft-vs-omit/{evaluation_metric}.json", "w") as evaluation_trends_file:
            json.dump(ft_vs_omit_evaluation_diffs[evaluation_metric], evaluation_trends_file)

        layer_trends.plot_layer_data(
            f"{evaluation_metric} Improvement\nKeep-Layer Ablation Model Against Base Model Performance",
            "Average Gain of Ablated Model Over Base Model",
            base_vs_keep_evaluation_diffs[evaluation_metric],
            f"results/ablation-analysis/layer/base-vs-keep/{evaluation_metric}.png"
        )
        layer_trends.plot_layer_data(
            f"{evaluation_metric} Improvement\nOmit-Layer Ablation Model Against Base Model Performance",
            "Average Gain of Ablated Model Over Base Model",
            base_vs_omit_evaluation_diffs[evaluation_metric],
            f"results/ablation-analysis/layer/base-vs-omit/{evaluation_metric}.png"
        )
        layer_trends.plot_layer_data(
            f"{evaluation_metric} Improvement\nKeep-Layer Ablation Model Against Fine-Tuned Model Performance",
            "Average Gain of Ablated Model Over Fine-Tuned Model",
            ft_vs_keep_evaluation_diffs[evaluation_metric],
            f"results/ablation-analysis/layer/ft-vs-keep/{evaluation_metric}.png"
        )
        layer_trends.plot_layer_data(
            f"{evaluation_metric} Improvement\nOmit-Layer Ablation Model Against Fine-Tuned Model Performance",
            "Average Gain of Ablated Model Over Fine-Tuned Model",
            ft_vs_omit_evaluation_diffs[evaluation_metric],
            f"results/ablation-analysis/layer/ft-vs-omit/{evaluation_metric}.png"
        )

def analyze_window_ablations():
    window_sizes = [ 2, 3, 4, 6 ]

    base_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/base-model.json")
    ft_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/ft-model.json")

    for window_size in window_sizes:
        base_vs_keep_evaluation_diffs = {
            evaluation_metric: []
            for evaluation_metric in evaluation_metrics.keys()
        }
        base_vs_omit_evaluation_diffs = {
            evaluation_metric: []
            for evaluation_metric in evaluation_metrics.keys()
        }
        ft_vs_keep_evaluation_diffs = {
            evaluation_metric: []
            for evaluation_metric in evaluation_metrics.keys()
        }
        ft_vs_omit_evaluation_diffs = {
            evaluation_metric: []
            for evaluation_metric in evaluation_metrics.keys()
        }

        for window_index in range(reranker.ft_model.config.num_hidden_layers - window_size + 1):
            keep_window_evaluation_scores = analyze_ablation(f"results/ablation-data/keep-window/window{window_index}-size{window_size}.json")
            omit_window_evaluation_scores = analyze_ablation(f"results/ablation-data/omit-window/window{window_index}-size{window_size}.json")

            for evaluation_metric in evaluation_metrics.keys():
                base_vs_keep_evaluation_diffs[evaluation_metric].append(keep_window_evaluation_scores[evaluation_metric] - base_model_evaluation_scores[evaluation_metric])
                base_vs_omit_evaluation_diffs[evaluation_metric].append(omit_window_evaluation_scores[evaluation_metric] - base_model_evaluation_scores[evaluation_metric])
                ft_vs_keep_evaluation_diffs[evaluation_metric].append(keep_window_evaluation_scores[evaluation_metric] - ft_model_evaluation_scores[evaluation_metric])
                ft_vs_omit_evaluation_diffs[evaluation_metric].append(omit_window_evaluation_scores[evaluation_metric] - ft_model_evaluation_scores[evaluation_metric])

        os.makedirs(f"results/ablation-analysis/window/size{window_size}/base-vs-keep", exist_ok=True)
        os.makedirs(f"results/ablation-analysis/window/size{window_size}/base-vs-omit", exist_ok=True)
        os.makedirs(f"results/ablation-analysis/window/size{window_size}/ft-vs-keep", exist_ok=True)
        os.makedirs(f"results/ablation-analysis/window/size{window_size}/ft-vs-omit", exist_ok=True)
        
        for evaluation_metric in evaluation_metrics.keys():
            with open(f"results/ablation-analysis/window/size{window_size}/base-vs-keep/{evaluation_metric}.json", "w") as evaluation_trends_file:
                json.dump(base_vs_keep_evaluation_diffs[evaluation_metric], evaluation_trends_file)
            with open(f"results/ablation-analysis/window/size{window_size}/base-vs-omit/{evaluation_metric}.json", "w") as evaluation_trends_file:
                json.dump(base_vs_omit_evaluation_diffs[evaluation_metric], evaluation_trends_file)
            with open(f"results/ablation-analysis/window/size{window_size}/ft-vs-keep/{evaluation_metric}.json", "w") as evaluation_trends_file:
                json.dump(ft_vs_keep_evaluation_diffs[evaluation_metric], evaluation_trends_file)
            with open(f"results/ablation-analysis/window/size{window_size}/ft-vs-omit/{evaluation_metric}.json", "w") as evaluation_trends_file:
                json.dump(ft_vs_omit_evaluation_diffs[evaluation_metric], evaluation_trends_file)

            layer_trends.plot_layer_data(
                f"{evaluation_metric} Improvement\nKeep-Window Ablation Model Against Base Model Performance",
                "Average Gain of Ablated Model Over Base Model",
                base_vs_keep_evaluation_diffs[evaluation_metric],
                f"results/ablation-analysis/window/size{window_size}/base-vs-keep/{evaluation_metric}.png",
                window_size=window_size
            )
            layer_trends.plot_layer_data(
                f"{evaluation_metric} Improvement\nOmit-Window Ablation Model Against Base Model Performance",
                "Average Gain of Ablated Model Over Base Model",
                base_vs_omit_evaluation_diffs[evaluation_metric],
                f"results/ablation-analysis/window/size{window_size}/base-vs-omit/{evaluation_metric}.png",
                window_size=window_size
            )
            layer_trends.plot_layer_data(
                f"{evaluation_metric} Improvement\nKeep-Window Ablation Model Against Fine-Tuned Model Performance",
                "Average Gain of Ablated Model Over Fine-Tuned Model",
                ft_vs_keep_evaluation_diffs[evaluation_metric],
                f"results/ablation-analysis/window/size{window_size}/ft-vs-keep/{evaluation_metric}.png",
                window_size=window_size
            )
            layer_trends.plot_layer_data(
                f"{evaluation_metric} Improvement\nOmit-Window Ablation Model Against Fine-Tuned Model Performance",
                "Average Gain of Ablated Model Over Fine-Tuned Model",
                ft_vs_omit_evaluation_diffs[evaluation_metric],
                f"results/ablation-analysis/window/size{window_size}/ft-vs-omit/{evaluation_metric}.png",
                window_size=window_size
            )

if __name__ == "__main__":
    base_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/base-model.json")
    ft_model_evaluation_scores = analyze_ablation("results/ablation-data/control/head/ft-model.json")

    print(base_model_evaluation_scores)
    print(ft_model_evaluation_scores)

    analyze_head_ablations()
    analyze_layer_ablations()
    analyze_window_ablations()