import os
import ir_datasets
from itertools import islice
import random
import json
import reranker

evaluation_collection = ir_datasets.load("msmarco-passage/dev")

def evaluate_model(reranker_model, num_queries: int, nrel_docs_per_query: int, output_path: str):
    documents_store = evaluation_collection.docs_store()
    queries = { query.query_id: query.text for query in evaluation_collection.queries_iter() }

    logit_groups = []

    for qrel in islice(evaluation_collection.qrels_iter(), num_queries):
        random.seed(qrel.query_id)

        logits = []

        documents = [
            documents_store.get(qrel.doc_id).text,
            *[
                documents_store.get(str(random.randint(0, 1000000))).text
                for _ in range(nrel_docs_per_query)
            ]
        ]

        while len(documents) > 0:
            logits = [
                *logits,
                *reranker.use_model(reranker_model, queries[qrel.query_id], documents[:10])
            ]

            documents = documents[10:]

        logit_groups.append({
            "rel": logits[0],
            "nrel": logits[1:]
        })

    with open(output_path, "w") as output_file:
        json.dump(logit_groups, output_file, indent=4)

if __name__ == "__main__":
    per_head = (50, 9) # 50q * 10d
    per_layer = (50, 99) # 50q * 100d
    per_window = (50, 99) # 50q * 100d

    print("Controls")

    os.makedirs("results/ablation-data/control/head", exist_ok=True)
    os.makedirs("results/ablation-data/control/layer", exist_ok=True)
    os.makedirs("results/ablation-data/control/window", exist_ok=True)

    with reranker.using_device(reranker.base_model):
        evaluate_model(reranker.base_model, per_head[0], per_head[1], "results/ablation-data/control/head/base-model.json")
        evaluate_model(reranker.base_model, per_layer[0], per_layer[1], "results/ablation-data/control/layer/base-model.json")
        evaluate_model(reranker.base_model, per_window[0], per_window[1], "results/ablation-data/control/window/base-model.json")

    with reranker.using_device(reranker.ft_model):
        evaluate_model(reranker.ft_model, per_head[0], per_head[1], "results/ablation-data/control/head/ft-model.json")
        evaluate_model(reranker.ft_model, per_layer[0], per_layer[1], "results/ablation-data/control/layer/ft-model.json")
        evaluate_model(reranker.ft_model, per_window[0], per_window[1], "results/ablation-data/control/window/ft-model.json")

        print("Heads")

        os.makedirs("results/ablation-data/omit-head", exist_ok=True)
        os.makedirs("results/ablation-data/keep-head", exist_ok=True)

        for layer_index in range(reranker.ft_model.config.num_hidden_layers):
            for head_index in range(reranker.ft_model.config.num_attention_heads):
                print(f"\tOmit Layer {layer_index}, Head {head_index}")

                with reranker.use_lora_ablated_model([ (layer_index, head_index) ]) as ablated_model:
                    evaluate_model(ablated_model, per_head[0], per_head[1], f"results/ablation-data/omit-head/layer{layer_index}-head{head_index}.json")

                print(f"\tKeep Layer {layer_index}, Head {head_index}")

                with reranker.use_lora_ablated_model([
                    (other_layer_index, other_head_index)
                    for other_head_index in range(reranker.ft_model.config.num_attention_heads)
                    for other_layer_index in range(reranker.ft_model.config.num_hidden_layers)
                    if not (other_layer_index == layer_index and other_head_index == head_index)
                ]) as ablated_model:
                    evaluate_model(ablated_model, per_head[0], per_head[1], f"results/ablation-data/keep-head/layer{layer_index}-head{head_index}.json")

        print("Layers")

        os.makedirs("results/ablation-data/omit-layer", exist_ok=True)
        os.makedirs("results/ablation-data/keep-layer", exist_ok=True)

        for layer_index in range(reranker.ft_model.config.num_hidden_layers):
            print(f"\tOmit Layer {layer_index}")

            with reranker.use_lora_ablated_model([
                (layer_index, head_index)
                for head_index in range(reranker.ft_model.config.num_attention_heads)
            ]) as ablated_model:
                evaluate_model(ablated_model, per_layer[0], per_layer[1], f"results/ablation-data/omit-layer/layer{layer_index}.json")

            print(f"\tKeep Layer {layer_index}")

            with reranker.use_lora_ablated_model([
                (other_layer_index, head_index)
                for head_index in range(reranker.ft_model.config.num_attention_heads)
                for other_layer_index in range(reranker.ft_model.config.num_hidden_layers)
                if other_layer_index != layer_index
            ]) as ablated_model:
                evaluate_model(ablated_model, per_layer[0], per_layer[1], f"results/ablation-data/keep-layer/layer{layer_index}.json")

        os.makedirs("results/ablation-data/omit-window", exist_ok=True)
        os.makedirs("results/ablation-data/keep-window", exist_ok=True)

        window_sizes = [ 2, 3, 4, 6 ]

        for window_size in window_sizes:
            print(f"Window Size {window_size}")

            for window_index in range(reranker.ft_model.config.num_hidden_layers - window_size + 1):
                print(f"\tOmit Window {window_index}")

                with reranker.use_lora_ablated_model([
                    (layer_index, head_index)
                    for head_index in range(reranker.ft_model.config.num_attention_heads)
                    for layer_index in range(window_index, window_index + window_size)
                ]) as ablated_model:
                    evaluate_model(ablated_model, per_window[0], per_window[1], f"results/ablation-data/omit-window/window{window_index}-size{window_size}.json")

                print(f"\tKeep Window {window_index}")

                with reranker.use_lora_ablated_model([
                    (layer_index, head_index)
                    for head_index in range(reranker.ft_model.config.num_attention_heads)
                    for layer_index in range(reranker.ft_model.config.num_hidden_layers)
                    if layer_index < window_index or layer_index >= window_index + window_size
                ]) as ablated_model:
                    evaluate_model(ablated_model, per_window[0], per_window[1], f"results/ablation-data/keep-window/window{window_index}-size{window_size}.json")