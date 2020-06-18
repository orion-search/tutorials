import numpy as np
import altair as alt


def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level 
    DistilBERT model and finds similar vectors using FAISS.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2abstract(df, I):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.id == idx]["original_title"]) for idx in I[0]]


def plot(data):
    """Plots with altair."""
    return (
        alt.Chart(data)
        .mark_circle(size=20)
        .encode(alt.X("Component 1"), alt.Y("Component 2"), tooltip=["title"])
        .interactive()
        .properties(width=650, height=500)
    )
