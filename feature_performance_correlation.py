import os
import json
import math
from scipy.stats import spearmanr

if __name__ == "__main__":
    model_data_paths = [ "base-model", "ft-model", "model-diffs" ]

    ablation_categories = [
        ("omit-layer", "layer"),
        ("omit-window/size2", "window-size2"),
        ("omit-window/size3", "window-size3"),
        ("omit-window/size4", "window-size4"),
        ("omit-window/size6", "window-size6"),
        ("keep-layer", "layer"),
        ("keep-window/size2", "window-size2"),
        ("keep-window/size3", "window-size3"),
        ("keep-window/size4", "window-size4"),
        ("keep-window/size6", "window-size6"),
    ]

    evaluation_metrics = [ "Mean Score Margin", "Min Score Margin", "Categorical Cross Entropy", "NDCG" ]

    for model_data_path in model_data_paths:
        for ablation_analysis_path, model_data_name in ablation_categories:
            os.makedirs(f"results/correlation/{model_data_path}/{ablation_analysis_path}", exist_ok=True)

            for evaluation_metric in evaluation_metrics:
                feature_correlation_table = {}

                for feature_name in os.listdir(f"results/{model_data_path}/layers"):
                    with open(f"results/ablation-analysis/{ablation_analysis_path}/{evaluation_metric}.json", "r") as ablation_analysis_file:
                        with open(f"results/{model_data_path}/layers/{feature_name}/{model_data_name}.json", "r") as model_data_file:
                            ablation_analysis_data = json.load(ablation_analysis_file)
                            model_data = json.load(model_data_file)

                            corr, _ = spearmanr(ablation_analysis_data, model_data)

                            if not math.isnan(corr):
                                feature_correlation_table[feature_name] = corr

                with open(f"results/correlation/{model_data_path}/{ablation_analysis_path}/{evaluation_metric}.json", "w") as correlation_data_file:
                    json.dump(dict(sorted(feature_correlation_table.items(), key = lambda entry: -abs(entry[1]))), correlation_data_file, indent=4)