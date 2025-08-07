import numpy as np
from typing import Optional

class Hypothesis:
    def __init__(self, idx, binval, parents=None, vector=None, n_bits=None):
        """
        idx: intero, serve per indicizzare la colonna di 'matrix' se è un singleton
        parents: lista di Hypothesis, i predecessori immediati di h
        """
        self.idx = idx
        self.parents = parents or []
        self.binval = binval
        self.n_bits = n_bits
        self.vector = vector

def bin_value_from_array(array: np.array):
    return ''.join(str(b) for b in array)

def bin_value(h: Hypothesis):
    return ''.join(str(b) for b in h.vector)

def preprocess_matrix_to_columns(matrix_file):
    """
    Legge la matrice per righe dal file e restituisce una lista di colonne (trasposta).
    """
    row_list = []
    with open(matrix_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;;"):
                continue
            if '-' in line:
                line = line.replace('-', '')
            row = [int(x) for x in line.split()]
            row_list.append(row)

    matrix = np.array(row_list, dtype=int)
    return matrix.T  # restituisce la matrice trasposta (per colonne)

def set_fields(h: Hypothesis, transposed_matrix: np.ndarray):
    # Inizializza il vettore a zero
    h.vector = np.zeros(transposed_matrix.shape[1], dtype=int)
    
    # Se non è l'ipotesi vuota
    if h.binval != "0":
        # Trova tutte le posizioni degli '1' (leggendo da sinistra a destra)
        for pos, bit in enumerate(h.binval):
            if bit == '1':
                h.vector[pos] = 1
            else:
                h.vector[pos] = 0
    
    return h

def create_currents(h0: Hypothesis, M):
    current = find_successors(h0, M)
    return current

def find_successors(h0: Hypothesis, M):
    if np.array_equal(h0.vector, np.ones(M, dtype=int)):
        return None
    else:
        successors = complement_zero_vectors(h0)
    return successors

def find_predecessors(h0: Hypothesis, M):
    if np.array_equal(h0.vector, np.zeros(M, dtype=int)):  # Corretto: predecessori del vettore zero
        return None
    else:
        predecessors = complement_one_vectors(h0)
    return predecessors

def complement_zero_vectors(h0):
    new_hs = []
    print(h0.vector)
    for i in range(len(h0.vector)):
        if h0.vector[i] == 0:
            nuovo_vettore = h0.vector.copy()
            nuovo_vettore[i] = 1  # complementiamo solo lo 0
            h = Hypothesis(idx=None, binval=bin_value_from_array(nuovo_vettore),
                         vector=nuovo_vettore, parents=[h0], n_bits=h0.n_bits)  # Corretto: parents è una lista
            new_hs.append(h)
    return new_hs

def complement_one_vectors(h0):
    new_hs = []
    for i in range(len(h0.vector)):
        if h0.vector[i] == 1:
            nuovo_vettore = h0.vector.copy()
            nuovo_vettore[i] = 0  # complementiamo solo lo 1
            h = Hypothesis(idx=None, binval=bin_value_from_array(nuovo_vettore),
                         vector=nuovo_vettore, parents=[h0], n_bits=h0.n_bits)  # Corretto: parents è una lista
            new_hs.append(h)
    return new_hs

def propagate(h: Hypothesis, h_prime: Hypothesis):
    """
    Propaga i bit da h a h_prime (successore),
    facendo OR bit‑a‑bit.
    """
    # h deve essere un predecessore immediato di h_prime
    h_prime.vector = np.bitwise_or(h_prime.vector, h.vector)

def check(h: Hypothesis) -> bool:
    """
    Ritorna True se in vector(h) non c'è alcuno zero,
    cioè tutti gli elementi di N sono coperti.
    """
    print(h.binval)
    return np.all(h.vector != 0)

def LM1(vector):
    for i, b in enumerate(vector):
        if b == 1:
            return i + 1
    return len(vector)  # Corretto: se non trova 1, ritorna la lunghezza

def prox(hp: Hypothesis, current) -> Optional[Hypothesis]:  # Corretto: current può essere lista
    """
    Restituisce l'ipotesi che viene immediatamente dopo hp in `current`.
    Ritorna None se hp è l'ultima o se hp non è presente.
    """
    try:
        # Trova l'indice di hp
        idx = current.index(hp)
        # Se c'è un elemento successivo, lo restituisce
        if idx + 1 < len(current):
            return current[idx + 1]
        else:
            return None
    except ValueError:
        # hp non è in current
        return None

def generate_children(h, current, M):
    children = []

    if np.array_equal(h.vector, np.zeros(M, dtype=int)):
        for i in range(M):
            h_prime_vec = h.vector.copy()
            h_prime_vec[i] = 1
            h_prime = Hypothesis(i, bin_value_from_array(h_prime_vec), vector=h_prime_vec, n_bits=M)
            set_fields(h_prime, matrix)
            children.append(h_prime)
        return children

    if not current:  # Verifica che current non sia vuoto
        return children

    h_first = current[0]

    for i in range(1, LM1(h.vector)):
        h_prime_vec = h.vector.copy()  # Corretto: parti dal vettore di h
        if i < len(h_prime_vec):  # Verifica bounds
            h_prime_vec[i] = 1

            h_prime = Hypothesis(i, binval=bin_value_from_array(h_prime_vec),
                               vector=h_prime_vec, n_bits=M)
            set_fields(h_prime, matrix)
            propagate(h, h_prime)

            h_initial = initial(h, h_prime, M)
            h_final = final(h, h_prime, M)
            cont = 0

            current_h_first = h_first  # Usa una variabile locale

            # CORREZIONE PRINCIPALE: Controlla che h_first non sia None prima di usarlo
            while (current_h_first is not None and
                   ((h_initial and bin_value(current_h_first) <= bin_value(h_initial)) or
                    (h_final and bin_value(current_h_first) >= bin_value(h_final)))):

                if (dist(bin_value(current_h_first), bin_value(h_prime)) == 1 and
                    dist(bin_value(current_h_first), bin_value(h)) == 2):
                    propagate(current_h_first, h)
                    cont += 1

                current_h_first = prox(current_h_first, current)
                if current_h_first is None:  # Esci se raggiungi la fine
                    break

            if cont == card(h):  # Assumendo che card() sia definita altrove
                children.append(h_prime)

    return children

def card(h):
    """Calcola la cardinalità (numero di 1) nel vettore di h"""
    return np.sum(h.vector)

def dist(v1, v2):
    """Calcola la distanza di Hamming tra due stringhe binarie"""
    if isinstance(v1, str) and isinstance(v2, str):
        return sum(c1 != c2 for c1, c2 in zip(v1, v2))
    else:
        return np.sum(v1 != v2)

def find_rightmost_one_index(vector):
    for i in range(len(vector) - 1, -1, -1):
        if vector[i] == 1:
            return i
    return -1

def find_leftmost_one_index(vector):
    for i in range(len(vector)):
        if vector[i] == 1:
            return i
    return -1

def initial(h, h_prime, M):
    if h_prime.vector is None:
        return None
        
    # Trova il primo bit a 1 partendo da destra (meno significativo)
    rightmost_one_index = find_rightmost_one_index(h_prime.vector)

    if rightmost_one_index == -1:
        return None

    new_vector = h_prime.vector.copy()
    new_vector[rightmost_one_index] = 0  # spegne il primo '1' da destra

    h_new = Hypothesis(0, bin_value_from_array(new_vector), vector=new_vector, n_bits=M)
    h_new.parents = find_predecessors(h_new, M)

    return h_new

def final(h, h_prime, M):
    # Trova il primo '1' più a sinistra e complementalo
    if h_prime.vector is None:
        return None

    leftmost_one_index = find_leftmost_one_index(h_prime.vector)
    if leftmost_one_index == -1:
        return None

    # Crea un nuovo vettore complementando il bit trovato
    new_vector = h_prime.vector.copy()
    new_vector[leftmost_one_index] = 0

    h_new = Hypothesis(0, bin_value_from_array(new_vector), vector=new_vector, n_bits=M)
    h_new.parents = find_predecessors(h_new, M)

    return h_new

def global_initial(h: Hypothesis, M) -> Hypothesis:
    n_bits = M.shape[1]
    bit_array = [int(b) for b in h.vector]  # Corretto: usa il vettore invece di binval

    if bit_array[0] == 0:
        bit_array[0] = 1
    else:
        print("bin(h)[1] =/= 0, non posso andare avanti.")
        return h

    # Step 2: Complementare il bit meno significativo impostato a 1
    for i in range(len(bit_array) - 1, -1, -1):
        if bit_array[i] == 1:
            bit_array[i] = 0
            break
    
    h = Hypothesis(idx=None, vector=np.array(bit_array),
                     binval=bin_value_from_array(bit_array), n_bits=n_bits)
    h = set_fields(h,M)
    #h.parents=find_predecessors(h,M)
    
    return Hypothesis(idx=None, vector=np.array(bit_array),
                     binval=bin_value_from_array(bit_array), n_bits=n_bits)

def remove_hr_from_current(current, h_second):
    ref_value = bin_value(h_second)
    
    # Filtra le ipotesi mantenendo solo quelle >=
    return [hr for hr in current if bin_value(hr) >= ref_value]

def merge(next_list, children):
    """
    Fonde `children` dentro `next_list`, mantenendo l'ordine decrescente
    di `hp.binval`.
    """
    # Semplificata per lavorare con liste
    next_list.extend(children)
    # Ordina per binval decrescente
    next_list.sort(key=lambda h: h.binval, reverse=True)
    return next_list

# Inizializzazione
matrix = np.array([
    [1,1,0,1,1,0,1],
    [0,0,1,1,0,1,0],
    [0,1,0,0,1,0,0]
]).T

print("Matrice di colonne:")
print(matrix)
print("\n")

n_bits = matrix.shape[1]
h0 = Hypothesis(0, bin_value_from_array(np.zeros(n_bits, dtype=int)), n_bits=n_bits)  # Ipotesi vuota
h0 = set_fields(h0, matrix)  # vector = [0,0,0]

current = create_currents(h0, n_bits)

print("ipotesi iniziali trovate.", len(current))

for h in current:
    h = set_fields(h,matrix)
    print(h.binval)
    
soluzioni = []

while current:
    next_list = []
    for h in current[:]:  # Copia la lista per evitare modifiche durante l'iterazione
        if check(h):
            soluzioni.append(h)
            current.remove(h)
        elif all(bit == '0' for bit in h.binval):
            children = generate_children(h, current, n_bits)
            print("children? ", children)
            next_list.extend(children)
        elif LM1(h.binval) != '0':
            h_second = global_initial(h, matrix)
            current = remove_hr_from_current(current, h_second)
            
            if current:  # Verifica che current non sia vuoto
                h_prime = current[0]
                if h_prime != h:
                    children = generate_children(h, current, n_bits)
                    merge(next_list, children)
        
    current = next_list

print(f"Trovate {len(soluzioni)} soluzioni:")