# The Emergence of Relevance Through Axiomatic Attention Patterns During LoRA Fine-Tuning

## Installation

All libraries are available in *requirements.txt*.

## Results

All figure and table data is located in the *results* directory.

## Reproducibility Files

**Ablation Data**: *evaluate_model.py*.

**Ablation Performance**: *analyze_ablations.py*.

**Attention Data**: *poll_attention_mass.py*.

**Attention Deltas and Heatmaps**: *calculate_model_diffs.py*.

**Attention Graphs**: *layer_trends.py*.

**Ablation-Attention Correlations**: *attention_performance_correlation.py*.

## Helper Scripts

**RankLLaMA Setup**: *reranker.py*.

**MS MARCO Collection Setup**: *collection_statistics.py*.

**Identifying Feature Pairs**: *tag.py*.

**Individual and Compositional Feature Definitions**: *attention_features.py*.

## Slurm Job Helpers

*salloc* and *sbatch.sh* contain helpful templates for managing GPU jobs.