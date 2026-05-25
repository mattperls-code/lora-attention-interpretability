import ir_datasets
import sys
from pathlib import Path
from collections import Counter
from itertools import islice
import spacy
import reranker
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

collection = ir_datasets.load("msmarco-passage/train")

if (
    Path(sys.modules['__main__'].__file__).name not in [
        "calculate_model_diffs.py",
        "evaluate_model.py",
        "analyze_ablations.py",
        "causality.py"
    ]
):
    
    use_collection_subset = True
    subset_collection_size = 500000
    BATCH_SIZE = 1024

    collection_doc_count = subset_collection_size if use_collection_subset else collection.docs_count()
    collection_word_count = 0
    collection_token_count = 0

    word_doc_freq = Counter()
    word_occurrences = Counter()
    token_doc_freq = Counter()
    token_occurrences = Counter()

    word_tokenizer = spacy.blank("en")
    doc_iter = islice(collection.docs_iter(), subset_collection_size) if use_collection_subset else collection.docs_iter()

    def batched(iterable, n):
        while batch := list(islice(iter(iterable), n)):
            yield batch

    print("Starting Collection Statistics Calculation")

    for batch in tqdm(batched(doc_iter, BATCH_SIZE), total=collection_doc_count // BATCH_SIZE):
        texts = [ document.text for document in batch ]

        # spacy batch
        for spacy_document in word_tokenizer.pipe(texts, batch_size=BATCH_SIZE):
            document_words = [ word.text.lower() for word in spacy_document ]
            word_doc_freq.update(set(document_words))
            word_occurrences.update(document_words)
            collection_word_count += len(document_words)

        # reranker tokenizer batch
        batch_encodings = reranker.tokenizer(texts, truncation=False, add_special_tokens=False)
        for input_ids in batch_encodings.input_ids:
            token_doc_freq.update(set(input_ids))
            token_occurrences.update(input_ids)
            collection_token_count += len(input_ids)

    print("Finished Collection Statistics Calculation")

    # words appearing in bottom 1% of idf are common, rest are rare

    idf_values = [ math.log(collection_doc_count / (word_freq + 1)) for word_freq in word_doc_freq.values() ]
    idf_values.sort()

    empirical_rarity_threshold = 182
    empirical_idf_theshold = idf_values[empirical_rarity_threshold]

    def idf_range(score: float):
        return "low" if score < math.log(1000 / 25) else "high"

            # OLD

            # # rough bounds for idf and ido ranges

            # very_low_idf = math.log(1000 / 200) # 200 in every thousand docs
            # low_idf = math.log(1000 / 100) # 100 in every thousand docs
            # med_idf = math.log(1000 / 25) # 25 in every thousand docs
            # high_idf = math.log(1000 / 2) # 2 in every thousand docs

            # def idf_range(score: float):
            #     if score < very_low_idf: return "very low"
            #     if score < low_idf: return "low"
            #     if score < med_idf: return "med"
            #     if score < high_idf: return "high"
            #     else: return "very high"

    # TODO: fix, but not used anywhere anyway

    # uses idf ranges, assuming 25 words per doc
    very_low_ido = math.log(25 * 1000 / 200) # 200 in every 25 thousand words
    low_ido = math.log(25 * 1000 / 100) # 100 in every 25 thousand words
    med_ido = math.log(25 * 1000 / 25) # 25 in every 25 thousand words
    high_ido = math.log(25 * 1000 / 2) # 2 in every 25 thousand words

    def ido_range(score: float):
        if score < very_low_ido: return "very low"
        if score < low_ido: return "low"
        if score < med_ido: return "med"
        if score < high_ido: return "high"
        else: return "very high"

if __name__ == "__main__":
    percent_values = [ 100 * idf_index / len(idf_values) for idf_index in range(len(idf_values)) ]

    frac_thresholds = [ 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20 ]

    for frac_threshold in frac_thresholds:
        threshold_index = int(len(idf_values) * frac_threshold / 100)

        print(f"{frac_threshold}% of words have idf below {idf_values[threshold_index]}")

    print(f"total num words: {len(idf_values)}")
    print(f"index of old 25/1000 docs: {next(i for i, idf in enumerate(idf_values) if idf >= math.log(1000/25))}")

    idf_threshold_index = next(i for i, idf in enumerate(idf_values) if idf >= empirical_idf_theshold)
    percent_rarer_than_threshold = 100 * (len(idf_values) - idf_threshold_index) / len(idf_values)
    
    plt.clf()
    plt.plot(idf_values, percent_values)
    plt.title("Empirical Metrics For Word Rarity", pad=10)
    plt.xlabel("Inverse Document Frequency")
    plt.ylabel("Percent Of Words Below IDF")
    plt.axvline(x=empirical_idf_theshold, color='red', linestyle=':', label=f'Empirical Rare Word Threshold')
    plt.axhline(y=100 - percent_rarer_than_threshold, color='black', linestyle=':', alpha=0.2)
    plt.annotate(
        f'{percent_rarer_than_threshold:.3f}% of Words Are More Rare',
        xy=(empirical_idf_theshold, 100 - percent_rarer_than_threshold),
        xytext=(empirical_idf_theshold + 0.5, 100 - percent_rarer_than_threshold + 8),
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red',
        ha='left'
    )
    plt.legend()
    plt.savefig("results/empirical-rarity-threshold.png")
    plt.close()