# vector.pyx
from cython.parallel import parallel, prange
cimport numpy as np  # cimport para numpy
import numpy as np

# Função para multiplicar um vetor por um escalar
def vector_by_scalar(np.ndarray[float, ndim=1] vector, double scalar):
    cdef int i
    cdef int n = len(vector)
    
    # Usando OpenMP para paralelismo com GIL liberado
    with parallel():
        with nogil:  # Liberar o GIL para permitir execução paralela
            for i in prange(n):  # prange permite a execução paralela
                vector[i] *= scalar
