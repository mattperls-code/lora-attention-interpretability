import ir_datasets
import sys
from pathlib import Path
from collections import Counter
from itertools import islice
import spacy
import reranker
import math
from tqdm import tqdm

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

    # rough bounds for idf and ido ranges

    very_low_idf = math.log(1000 / 200) # 200 in every thousand docs
    low_idf = math.log(1000 / 100) # 100 in every thousand docs
    med_idf = math.log(1000 / 25) # 25 in every thousand docs
    high_idf = math.log(1000 / 2) # 2 in every thousand docs

    def idf_range(score: float):
        if score < very_low_idf: return "very low"
        if score < low_idf: return "low"
        if score < med_idf: return "med"
        if score < high_idf: return "high"
        else: return "very high"

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