from typing import Callable
from collections import defaultdict
import string
import spacy
import reranker
import torch
from transformers import AutoTokenizer, AutoModel
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
import collection_statistics
import math

class TaggedToken:
    id: int
    start: int
    end: int

    categorical_tags: dict[str, str]
    numeric_tags: dict[str, float]
    other_tags: dict

    def __init__(self, index: int, id: int, start: int, end: int, text: str):
        self.index = index
        self.id = id
        self.start = start
        self.end = end
        self.text = text

        self.categorical_tags = defaultdict(lambda: "")
        self.numeric_tags = defaultdict(lambda: 0)
        self.other_tags = {}

    def __str__(self):
        categorical_tags_str = ", ".join(f"{k}=\"{v}\"" for k, v in self.categorical_tags.items())
        numerical_tags_str = ", ".join(f"{k}={v}" for k, v in self.numeric_tags.items())

        tags = ", ".join([ categorical_tags_str, numerical_tags_str ])

        return f"{{ text=\"{self.text}\"" + (f", {tags}" if tags else "") + " }"
    
def tokenize(text: str, start_index: int):
    tokens = reranker.tokenizer(text, return_offsets_mapping=True)
    
    return [ TaggedToken(index + start_index, id, start, end, text[start : end]) for index, (id, (start, end)) in enumerate(zip(tokens["input_ids"], tokens["offset_mapping"])) ]

# https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl
pos_tagger = spacy.load("en_core_web_trf", disable=["parser", "ner", "lemmatizer"])

def tag_pos(tagged_tokens: list[TaggedToken], text: str):
    word_list = pos_tagger(text)

    for tagged_token in tagged_tokens:
        if tagged_token.start == tagged_token.end: continue

        # llama tokens often includes spaces, adjust accordingly so its not artificially outside the word bounds
        tagged_token_start = tagged_token.start + (len(tagged_token.text) - len(tagged_token.text.lstrip(string.whitespace)))
        tagged_token_end = tagged_token.end - (len(tagged_token.text) - len(tagged_token.text.rstrip(string.whitespace)))

        tagged_token.numeric_tags["start_index"] = tagged_token_start
        tagged_token.numeric_tags["end_index"] = tagged_token_end

        for word_index, word in enumerate(word_list):
            if tagged_token_start >= word.idx and tagged_token_end <= word.idx + len(word.text):
                tagged_token.categorical_tags["pos"] = word.pos_
                tagged_token.categorical_tags["word"] = word.text
                tagged_token.numeric_tags["word_index"] = word_index

stopword_set = set(stopwords.words("english"))

# call after tagging pos to extract word
def tag_stopword(tagged_tokens: list[TaggedToken], text: str):
    for tagged_token in tagged_tokens:
        tagged_token.other_tags["stopword"] = tagged_token.categorical_tags["word"].lower().strip() in stopword_set

similarity_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
similarity_model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")
# similarity_model.to(reranker.device)
similarity_model.eval()

# call after tagging pos to extract word
def tag_embedding(tagged_tokens: list[TaggedToken], text: str):
    similarity_tokens = similarity_tokenizer(text, return_offsets_mapping=True, return_tensors='pt')
    
    with torch.no_grad():
        similarity_model_output = similarity_model(**similarity_tokens, output_hidden_states=True)

        averaged_contextual_embeddings = torch.stack(similarity_model_output.hidden_states[-8:-4]).mean(dim=0).squeeze(0)

        for tagged_token in tagged_tokens:
            for similarity_token_index, (similarity_token_start, similarity_token_end) in enumerate(similarity_tokens["offset_mapping"][0, :, :].tolist()):
                if tagged_token.numeric_tags["start_index"] >= similarity_token_start and tagged_token.numeric_tags["end_index"] <= similarity_token_end:
                    tagged_token.other_tags["embedding"] = averaged_contextual_embeddings[similarity_token_index, :]

# call after tagging pos to extract word
def tag_collection_stats(tagged_tokens: list[TaggedToken], text: str):
    for tagged_token in tagged_tokens:
        word = tagged_token.categorical_tags["word"].lower().strip()

        word_idf = math.log(collection_statistics.collection_doc_count / (collection_statistics.word_doc_freq.get(word, 0) + 1))
        word_ido = math.log(collection_statistics.collection_word_count / (collection_statistics.word_occurrences.get(word, 0) + 1))

        token_idf = math.log(collection_statistics.collection_doc_count / (collection_statistics.token_doc_freq.get(tagged_token.id, 0) + 1))
        token_ido = math.log(collection_statistics.collection_token_count / (collection_statistics.token_occurrences.get(tagged_token.id, 0) + 1))

        tagged_token.numeric_tags["word_idf"] = word_idf
        tagged_token.numeric_tags["word_ido"] = word_ido
        tagged_token.numeric_tags["token_idf"] = token_idf
        tagged_token.numeric_tags["token_ido"] = token_ido

        tagged_token.categorical_tags["word_idf_range"] = collection_statistics.idf_range(word_idf)
        tagged_token.categorical_tags["word_ido_range"] = collection_statistics.ido_range(word_ido)
        tagged_token.categorical_tags["token_idf_range"] = collection_statistics.idf_range(token_idf)
        tagged_token.categorical_tags["token_ido_range"] = collection_statistics.ido_range(token_ido)

def tag_query(tagged_tokens: list[TaggedToken], text: str):
    for tagged_token in tagged_tokens:
        tagged_token.categorical_tags["type"] = "query"

def tag_document(tagged_tokens: list[TaggedToken], text: str):
    for tagged_token in tagged_tokens:
        tagged_token.categorical_tags["type"] = "document"

def generate_tagged_tokens(text: str, tags: list[Callable[[list[TaggedToken], str], None]], start_index: int):
    tagged_tokens = tokenize(text, start_index)

    for tag in tags: tag(tagged_tokens, text)

    return tagged_tokens

def is_token(token_id: int):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.id == token_id
    
    return predicate

def is_pos(pos: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["pos"] in pos
    
    return predicate

def is_stopword(tagged_token: TaggedToken):
    return tagged_token.other_tags["stopword"]

def is_word_idf_range(idf: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["word_idf_range"] in idf
    
    return predicate

def is_word_ido_range(ido: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["word_ido_range"] in ido
    
    return predicate

def is_token_idf_range(idf: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["token_idf_range"] in idf
    
    return predicate

def is_token_ido_range(ido: set[str]):
    def predicate(tagged_token: TaggedToken):
        return tagged_token.categorical_tags["token_ido_range"] in ido
    
    return predicate

def is_document(tagged_token: TaggedToken):
    return tagged_token.categorical_tags["type"] == "document"

def is_query(tagged_token: TaggedToken):
    return tagged_token.categorical_tags["type"] == "query"

def is_not(predicate: Callable[[TaggedToken], bool]):
    def negated_predicate(tagged_token: TaggedToken):
        return not predicate(tagged_token)
    
    return negated_predicate

def token_satisfies_all(predicates: list[Callable[[TaggedToken], bool]]):
    def conjunctive_predicate(tagged_token: TaggedToken):
        for predicate in predicates:
            if not predicate(tagged_token): return False

        return True
    
    return conjunctive_predicate

def filter_first(predicate: Callable[[TaggedToken], bool]):
    def filter_first_with_predicate(tagged_tokens: list[TaggedToken], pairs: list[tuple[int, int]]):
        return filter(
            lambda pair: predicate(tagged_tokens[pair[0]]),
            pairs
        )
    
    return filter_first_with_predicate

def filter_second(predicate: Callable[[TaggedToken], bool]):
    def filter_second_with_predicate(tagged_tokens: list[TaggedToken], pairs: list[tuple[int, int]]):
        return filter(
            lambda pair: predicate(tagged_tokens[pair[1]]),
            pairs
        )
    
    return filter_second_with_predicate

def are_exact_token_match(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return first_tagged_token.id == second_tagged_token.id

def are_exact_word_match(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return first_tagged_token.categorical_tags["word"].lower().strip() == second_tagged_token.categorical_tags["word"].lower().strip()

def are_synonyms(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return (
        not is_stopword(first_tagged_token) and
        not is_stopword(second_tagged_token) and
        first_tagged_token.categorical_tags["pos"] != "PUNCT" and
        second_tagged_token.categorical_tags["pos"] != "PUNCT" and
        "embedding" in first_tagged_token.other_tags and
        "embedding" in second_tagged_token.other_tags and
        torch.nn.functional.cosine_similarity(
            first_tagged_token.other_tags["embedding"],
            second_tagged_token.other_tags["embedding"],
            dim=0
        ).item() > 0.5
    )

def are_mirror(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return first_tagged_token.index == second_tagged_token.index

def are_adjacent(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return abs(first_tagged_token.index - second_tagged_token.index) == 1

def are_neighbors(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return (
        abs(first_tagged_token.numeric_tags["word_index"] - second_tagged_token.numeric_tags["word_index"]) <= 2 and
        first_tagged_token.categorical_tags["type"] == second_tagged_token.categorical_tags["type"]
    )

def are_same_word_group(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
    return (
        first_tagged_token.numeric_tags["word_index"] == second_tagged_token.numeric_tags["word_index"] and
        first_tagged_token.categorical_tags["type"] == second_tagged_token.categorical_tags["type"]
    )

def are_not(predicate: Callable[[TaggedToken, TaggedToken], bool]):
    def negated_predicate(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
        return not predicate(first_tagged_token, second_tagged_token)
    
    return negated_predicate

def pair_satisfies_all(predicates: list[Callable[[TaggedToken, TaggedToken], bool]]):
    def conjunctive_predicate(first_tagged_token: TaggedToken, second_tagged_token: TaggedToken):
        for predicate in predicates:
            if not predicate(first_tagged_token, second_tagged_token): return False

        return True
    
    return conjunctive_predicate

def filter_combination(predicate: Callable[[TaggedToken, TaggedToken], bool]):
    def filter_combination_with_predicate(tagged_tokens: list[TaggedToken], pairs: list[tuple[int, int]]):
        return filter(
            lambda pair: predicate(tagged_tokens[pair[0]], tagged_tokens[pair[1]]),
            pairs
        )
    
    return filter_combination_with_predicate

def filter_tagged_token_pairs(tagged_tokens: list[TaggedToken], pair_filters: list[Callable[[list[TaggedToken], list[tuple[int, int]]], set[tuple[int, int]]]]):
    pairs = []

    for i in range(len(tagged_tokens)):
        for j in range(len(tagged_tokens)):
            pairs.append((i, j))

    for pair_filter in pair_filters:
        pairs = pair_filter(tagged_tokens, pairs)

    return set(pairs)

if __name__ == "__main__":
    text = "There are very many ways to talk about vocabulary. It can be seen as an individual persons vernacular, as a mapping of terms to definitions, or, in the context of information retrieval, as the set of unique word strings across some corpus. Ultimately, how people talk and communicate is a complicated process, and trying to capture the nature of speech remains a barrier to textual modeling and natural language processing."

    tagged_tokens = generate_tagged_tokens(text, [
        tag_pos,
        tag_stopword,
        tag_embedding,
        tag_collection_stats
    ], 0)

    print("Tokens:")

    for tagged_token in tagged_tokens: print(f"{tagged_token}\n")

    print()

    rare_words = [ tagged_token.categorical_tags["word"] for tagged_token in tagged_tokens if is_word_idf_range([ "high" ])(tagged_token) ]

    print("Rare Words")
    print(rare_words)

    very_rare_words = [ tagged_token.categorical_tags["word"] for tagged_token in tagged_tokens if is_word_idf_range([ "very high" ])(tagged_token) ]

    print("Very Rare Words")
    print(very_rare_words)

    print()

    synonymous_pairs = filter_tagged_token_pairs(tagged_tokens, [ filter_combination(are_synonyms) ])

    print("Synonymous Words:")

    for token_index1, token_index2 in synonymous_pairs:
        if tagged_tokens[token_index1].numeric_tags["word_index"] != tagged_tokens[token_index2].numeric_tags["word_index"]:
            print(f"{tagged_tokens[token_index1].categorical_tags["word"]} and {tagged_tokens[token_index2].categorical_tags["word"]}")