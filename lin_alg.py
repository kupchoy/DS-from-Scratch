from typing import List

Vector = List[float]
#%%
def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), 'vectors must be the same length'

    return [v_i + w_i for v_i, w_i in zip(v, w)]

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]
#%%
add([1, 2, 3], [4, 5, 6])

#%%
subtract([5, 7, 9], [4, 5, 6])
#%%
def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    assert vectors, 'no vectors provided'

    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), 'different sizes'

    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]

vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]])
#%%

