import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import tag
import reranker
import attention_features

def attention_heatmap(output_file: str, title: str, tagged_tokens: list[tag.TaggedToken], attention_matrix: torch.Tensor, pairs: set[tuple[int, int]]):
    scores = {}

    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            scores[( i, j )] = ((i, j) in pairs, attention_matrix[i, j].item())

    plt.clf()

    fig, ax = plt.subplots()

    x = 0.06

    tile_coords = []

    for tagged_token in tagged_tokens:
        horizontal_text_obj = ax.text(x, 0, tagged_token.text, fontsize=18, va="bottom", ha="left")
        
        ax.text(x - 0.02, x + 0.02, tagged_token.text, fontsize=18, va="bottom", ha="right", rotation=90)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        bbox_pixels = horizontal_text_obj.get_window_extent(renderer=renderer)
        x_pixels = ax.transData.transform([(x, 0)])[0][0]
        pixel_width = bbox_pixels.width
        x_end_pixels = x_pixels + pixel_width
        x_start_data = ax.transData.inverted().transform([(x_pixels, 0)])[0][0]
        x_end_data = ax.transData.inverted().transform([(x_end_pixels, 0)])[0][0]
        width = x_end_data - x_start_data

        tile_coords.append((x, width))

        x += width

    for attended_index in range(len(tagged_tokens)):
        for attending_index in range(attended_index, len(tagged_tokens)):
            if (attending_index, attended_index) in scores:
                is_feature, value = scores[(attending_index, attended_index)]

                color = (0.0, 0.0, value) if attended_index == 0 else (0.0, value, 0.0) if is_feature else (value, 0.0, 0.0)

                ax.add_patch(patches.Rectangle(
                    (tile_coords[attending_index][0], tile_coords[attended_index][0]),
                    tile_coords[attending_index][1],
                    tile_coords[attended_index][1],
                    linewidth=0,
                    facecolor=color
                ))

    unnormalized_attention_mass, normalized_attention_mass = attention_features.calculate_attention_mass(attention_matrix, attention_features.CompositeFeatureTable.build_feature_mask(
        pairs,
        attention_matrix.shape[-1]
    ))

    ax.text(0.06, x - 0.03, title, fontsize=18, fontweight="bold", va="top", ha="left")
    ax.text(
        0.06, x - 0.17,
        f"Unnormalized Feature Attention: {unnormalized_attention_mass:.3f}\nNormalized Feature Attention: {normalized_attention_mass:.3f}",
        fontsize=18, va="top", ha="left"
    )

    ax.legend(handles=[
        matplotlib.lines.Line2D([], [], color="none", label="X-Axis: Attending Token"),
        matplotlib.lines.Line2D([], [], color="none", label="Y-Axis: Attended Token"),
        patches.Patch(facecolor="green", label="Feature Attention"),
        patches.Patch(facecolor="red", label="Non-Feature Attention"),
        patches.Patch(facecolor="blue", label="Sink Attention"),
    ], fontsize=18, loc="upper left", bbox_to_anchor=(0.06, x - 0.3), bbox_transform=ax.transData)

    fig.canvas.draw()

    current_fig_width = fig.get_figwidth()
    current_xlim = ax.get_xlim()
    coefficient = current_fig_width / (current_xlim[1] - current_xlim[0])

    fig.set_size_inches(x * coefficient, x * coefficient)

    ax.set_xlim(0, x)
    ax.set_ylim(0, x)
    ax.axis("off")

    plt.savefig(f"{output_file}", dpi=100, bbox_inches="tight")

    plt.close()

def transformer_heatmap(output_file: str, title: str, head_data: list[list[float]]):
    plt.clf()

    num_attention_layers = len(head_data)
    num_attention_heads = len(head_data[0])

    cell_size = 0.25
    gap = 0.05

    red_white_green = mcolors.LinearSegmentedColormap.from_list("red_white_green", ["red", "white", "green"])
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, ax = plt.subplots(figsize=(num_attention_layers * (cell_size + gap) - gap, num_attention_heads * (cell_size + gap) - gap))
    
    for attention_layer_index in range(num_attention_layers):
        for attention_head_index in range(num_attention_heads):
            ax.add_patch(patches.Rectangle(
                (attention_layer_index * (cell_size + gap), attention_head_index * (cell_size + gap)),
                cell_size,
                cell_size,
                color=red_white_green(norm(head_data[attention_layer_index][attention_head_index]))
            ))

    ax.set_title(title, pad=15)
    ax.set_xlabel("Attention Layer", labelpad=10)
    ax.set_ylabel("Attention Head", labelpad=10)

    ax.set_xticks([col * (cell_size + gap) + 0.5 * cell_size for col in range(num_attention_layers)])
    ax.set_xticklabels(range(1, num_attention_layers + 1))
    ax.set_yticks([row * (cell_size + gap) + 0.5 * cell_size for row in range(num_attention_heads)])
    ax.set_yticklabels(range(1, num_attention_heads + 1))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_xlim(-gap, num_attention_layers * (cell_size + gap) - gap)
    ax.set_ylim(-gap, num_attention_heads * (cell_size + gap) - gap)

    sm = plt.cm.ScalarMappable(cmap=red_white_green, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.set_ticks([ -1, 0, 1 ])
    cbar.set_ticklabels([ "-1", "0", "1" ])

    ax.set_aspect("equal")
    plt.tight_layout()

    plt.savefig(output_file)

    plt.close()

def example_heatmap(model, tuning_name):
    with reranker.using_device(model) as reranker_model:
        query_text = "query: Are house cats the same species as lions?"
        document_text = "document: Lions, tigers, and panthers all belong to the same family."

        tagged_query_tokens = tag.generate_tagged_tokens(query_text, [
            tag.tag_query,
            tag.tag_pos,
            tag.tag_stopword,
            tag.tag_embedding,
            tag.tag_collection_stats
        ], 0)

        tagged_document_tokens = tag.generate_tagged_tokens(document_text, [
            tag.tag_document,
            tag.tag_pos,
            tag.tag_stopword,
            tag.tag_embedding,
            tag.tag_collection_stats
        ], len(tagged_query_tokens))

        all_tagged_tokens = tagged_query_tokens + tagged_document_tokens

        attention_layers = reranker.get_attention_layers(reranker_model, query_text, document_text)

        composite_feature_table = attention_features.CompositeFeatureTable(all_tagged_tokens)

        os.makedirs(f"./results/{tuning_name}/attention-heatmaps", exist_ok=True)

        model_name = "Base Model" if tuning_name == "base-model" else "Fine-Tuned Model"

        composite_feature1 = "Rare Tokens Attending Rare Semantic Match Tokens"
        attention_heatmap(f"results/{tuning_name}/attention-heatmaps/1.png", f"{composite_feature1}\n{model_name}, Layer 8, Head 6", all_tagged_tokens, attention_layers[8, 6, :, :], composite_feature_table.get(composite_feature1))

        composite_feature2 = "Very Rare Document Tokens Attending Query Tokens"
        attention_heatmap(f"results/{tuning_name}/attention-heatmaps/2.png", f"{composite_feature2}\n{model_name}, Layer 16, Head 17", all_tagged_tokens, attention_layers[16, 17, :, :], composite_feature_table.get(composite_feature2))

        composite_feature3 = "All Tokens Attending Rare Tokens"
        attention_heatmap(f"results/{tuning_name}/attention-heatmaps/3.png", f"{composite_feature3}\n{model_name}, Layer 8, Head 6", all_tagged_tokens, attention_layers[8, 6, :, :], composite_feature_table.get(composite_feature3))

        composite_feature4 = "All Tokens Attending Lexical Match Tokens"
        attention_heatmap(f"results/{tuning_name}/attention-heatmaps/4.png", f"{composite_feature4}\n{model_name}, Layer 16, Head 17", all_tagged_tokens, attention_layers[16, 17, :, :], composite_feature_table.get(composite_feature4))

if __name__ == "__main__":
    example_heatmap(reranker.base_model, "base-model")
    example_heatmap(reranker.ft_model, "ft-model")