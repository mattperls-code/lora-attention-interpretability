import os
import json
import random
from evaluate_model import evaluate_model
import reranker

random_trial_count = 10

features = [
    "All Tokens Attending Lexical Match Tokens",
    "All Tokens Attending Rare Tokens",
    "Document Tokens Attending Query Tokens"
]

k_values = [ 4, 8, 16, 32 ]

for k_value in k_values:
    os.makedirs(f"results/causal-ablations/random/k_{k_value}", exist_ok=True)
    
    for random_trial_index in range(random_trial_count):
        random_k_heads = random.sample([
            (i, j)
            for i in range(reranker.ft_model.config.num_hidden_layers)
            for j in range(reranker.ft_model.config.num_attention_heads)
        ], k_value)

        print(f"Ablating {k_value} random heads, trial {random_trial_index}")

        with reranker.use_lora_ablated_model(random_k_heads) as ablated_model:
            evaluate_model(ablated_model, 50, 99, f"results/causal-ablations/random/k_{k_value}/trial{random_trial_index}.json")

for feature in features:
    os.makedirs(f"results/causal-ablations/{feature}", exist_ok=True)

    with open(f"results/model-diffs/transformer-heatmaps/{feature}/normalized.json", "r") as ft_attention_increase_file:
        ft_attention_increase_data = json.load(ft_attention_increase_file)

        attn_increase_per_head = [
            (ft_attention_increase_data[i][j], i, j)
            for i in range(len(ft_attention_increase_data))
            for j in range(len(ft_attention_increase_data[i]))
        ]

        attn_increase_per_head.sort(key=lambda x: -x[0])

        for k_value in k_values:
            print(f"Ablating top {k_value} learned heads for {feature}")

            top_k_heads = [(i, j) for _, i, j in attn_increase_per_head[:k_value]]

            with reranker.use_lora_ablated_model(top_k_heads) as ablated_model:
                evaluate_model(ablated_model, 50, 99, f"results/causal-ablations/{feature}/k_{k_value}.json")