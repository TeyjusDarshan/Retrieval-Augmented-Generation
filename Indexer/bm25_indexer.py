from datasets import load_dataset
import bm25s
import Stemmer


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("rajpurkar/squad")


doc_list = list(ds['train']['context'])
doc_list.extend(list(ds['validation']['context']))

unique_docs = set(doc_list)
unique_docs = list(unique_docs)



stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(
    unique_docs, 
    stopwords="en", 
    stemmer=stemmer
)

print(corpus_tokens)


retriever = bm25s.BM25(method="lucene", corpus=unique_docs)

print("Baking BM25 scores into sparse matrix...")
retriever.index(corpus_tokens)

retriever.save("bm25_wikipedia_index", corpus=unique_docs)
print("Index successfully built and saved to disk!")



