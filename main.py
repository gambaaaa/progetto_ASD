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
    return ''.join(str(b) for b in h.binval)

def bin_value_to_int(binval: str) -> int:
    """
    Converte una stringa binaria (es: '010') nel corrispondente valore intero.
    """
    return int(binval, 2)

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
    n_rows = transposed_matrix.shape[0]  # Numero di righe (es: 7)
    h.vector = np.zeros(n_rows, dtype=int)
    if h.binval != "0":
        indices = [i for i, bit in enumerate(h.binval) if bit == '1']
        if len(indices) == 1:
            # Singoletto: copia la colonna corrispondente
            h.vector = transposed_matrix[:, indices[0]].copy()
        else:
            # OR bitwise tra tutte le colonne corrispondenti agli '1'
            for idx in indices:
                h.vector = np.bitwise_or(h.vector, transposed_matrix[:, idx])

    return h

def create_currents(h0: Hypothesis, M):
    current = find_successors(h0, M)
    return current

def find_successors(h0: Hypothesis, M):
    if h0.binval == '1' * M:
        return None
    else:
        successors = complement_zero_vectors(h0)
        
    print("Successori trovati:", [s.binval for s in successors])
    return successors

def find_predecessors(h0: Hypothesis, M):
    if h0.binval == '0' * M:
        return None
    else:
        predecessors = complement_one_vectors(h0)
    
    print("h0 binval:", h0.binval)
    print("Predecessori trovati:", [p.binval for p in predecessors])
    return predecessors

def complement_zero_vectors(h0):
    new_hs = []
    for i, bit in enumerate(h0.binval):
        if bit == '0':
            nuovo_binval = list(h0.binval)
            nuovo_binval[i] = '1'
            nuovo_binval_str = ''.join(nuovo_binval)
            h = Hypothesis(idx=None, binval=nuovo_binval_str,
                           n_bits=h0.n_bits, parents=[h0])
            new_hs.append(h)
    return new_hs

def complement_one_vectors(h0):
    new_hs = []
    for i, bit in enumerate(h0.binval):
        if bit == '1':
            nuovo_binval = list(h0.binval)
            nuovo_binval[i] = '0'
            nuovo_binval_str = ''.join(nuovo_binval)
            h = Hypothesis(idx=None, binval=nuovo_binval_str,
                           n_bits=h0.n_bits, parents=[h0])
            new_hs.append(h)
    return new_hs

def propagate(h: Hypothesis, h_prime: Hypothesis):
    """
    Propaga i bit da h a h_prime (successore),
    facendo OR bit‑a‑bit.
    """
    # h deve essere un predecessore immediato di h_prime
    vector = np.bitwise_or(h_prime.vector, h.vector)
    return vector

def check(h: Hypothesis) -> bool:
    """
    Ritorna True se in vector(h) non c'è alcuno zero,
    cioè tutti gli elementi di N sono coperti.
    """
    return np.all(h.vector != 0)

def LM1(binval: str):
    for i, b in enumerate(binval):
        if b == '1':
            return i
    return -1  # Se non trova '1', ritorna -1

def prox(hp: Hypothesis, current) -> Optional[Hypothesis]:
    """
    Restituisce l'ipotesi che viene immediatamente dopo hp in `current`
    (con lo stesso binval). Ritorna None se hp è l'ultima o non è presente.
    """
    for idx, hr in enumerate(current):
        if hr.binval == hp.binval:
            if idx + 1 < len(current):
                return current[idx + 1]
            else:
                return None
    return None

def generate_children(h, current, matrix):
    children = []
    
    if np.array_equal(h.vector, np.zeros(matrix.shape[1], dtype=int)):
        for i in range(matrix):
            h_prime_vec = h.vector.copy()
            h_prime_vec[i] = 1
            h_prime = Hypothesis(i, bin_value_from_array(h_prime_vec), n_bits=matrix.shape[1])
            set_fields(h_prime, matrix.shape[1])
            children.append(h_prime)
        return children

    if not current:  # Verifica che current non sia vuoto
        return children

    for i in range(0, LM1(h.binval)):
        # Crea h_prime
        h_prime_bin_list = [int(b) for b in h.binval]
        h_prime_bin_list[i] = 1
        h_prime = Hypothesis(i, binval=bin_value_from_array(h_prime_bin_list), n_bits=matrix.shape[1])

        set_fields(h_prime, matrix)
        h_prime.vector = propagate(h, h_prime)

        print("h_prime", h_prime.binval)
        h_initial = initial(h, h_prime, matrix)
        h_final = final(h, h_prime, matrix)

        print("h_initial", h_initial.binval)
        print("h_final", h_final.binval)
        cont = 0
        # Per ogni iterazione, parti dal primo elemento di current
        current_h_first = current[0]
        
        print("h_prime", h_prime.binval)
        while (current_h_first is not None and
               bin_value_to_int(current_h_first.binval) <= bin_value_to_int(h_initial.binval) and
               bin_value_to_int(current_h_first.binval) >= bin_value_to_int(h_final.binval)):
            
            print("Entra?")
            if (dist(bin_value(current_h_first), bin_value(h_prime)) == 1 and
                dist(bin_value(current_h_first), bin_value(h)) == 2):
                current_h_first.vector = propagate(current_h_first, h_prime)
                cont += 1

            current_h_first = prox(current_h_first, current)
        
        print("contatore = ", cont)
        
        if cont == card(h):
            print("PADRE : ", h.binval)
            print("È STATO AGGIUNTO = ", h_prime.binval)
            children.append(h_prime)
    
    print("FINE CICLO! -------------------")
    return children

def card(h):
    """Calcola la cardinalità (numero di '1') nella stringa binval di h"""
    return h.binval.count('1')

def dist(v1, v2):
    """Calcola la distanza di Hamming tra due stringhe binarie"""
    if isinstance(v1, str) and isinstance(v2, str):
        return sum(c1 != c2 for c1, c2 in zip(v1, v2))
    else:
        return np.sum(v1 != v2)

def find_initial(binval):
    """
    Riceve una stringa binaria, trasforma in '0' il primo '1' che incontra da destra,
    e restituisce la nuova stringa.
    """
    bin_list = list(binval)
    for i in range(len(bin_list) - 1, -1, -1):
        if bin_list[i] == '1':
            bin_list[i] = '0'
            break
    return ''.join(bin_list)

def find_final(bin_str):
    """
    Riceve una stringa binaria, trasforma in '0' il primo '1' che incontra da sinistra,
    e restituisce la nuova stringa.
    """
    bin_list = list(bin_str)
    for i in range(len(bin_list)):
        if bin_list[i] == '1':
            bin_list[i] = '0'
            break
    return ''.join(bin_list)

def initial(h: Hypothesis, h_prime: Hypothesis, M) -> Optional[Hypothesis]:
    """
    Trova il predecessore immediato più a sinistra di h_prime tra tutti i suoi
    predecessori immediati che sono distinti da h e posti a sinistra di h.
    """
    predecessors = find_predecessors(h_prime, M.shape[1])
    if not predecessors:
        return None

    # Filtra i predecessori: mantieni solo quelli a sinistra di h
    left_predecessors = [p for p in predecessors 
                        if p.binval != h.binval and 
                        bin_value_to_int(p.binval) > bin_value_to_int(h.binval)]
    
    if not left_predecessors:
        return None

    # Se ci sono solo due predecessori, prendi quello con valore binario più grande
    if len(left_predecessors) == 2:
        # Ordina per valore binario decrescente e prendi il primo (più grande)
        sorted_preds = sorted(left_predecessors, 
                            key=lambda x: bin_value_to_int(x.binval), 
                            reverse=True)
    else:
        leftmost = max(left_predecessors, key=lambda x: bin_value_to_int(x.binval))
    
    leftmost = set_fields(leftmost, M)
    return leftmost

def final(h: Hypothesis, h_prime: Hypothesis, M) -> Optional[Hypothesis]:
    """
    Trova il predecessore immediato più a destra di h_prime tra tutti i suoi
    predecessori immediati che sono distinti da h e posti a sinistra di h.
    """
    predecessors = find_predecessors(h_prime, M.shape[1])
    if not predecessors:
        return None

    # Filtra i predecessori: mantieni solo quelli a sinistra di h
    left_predecessors = [p for p in predecessors 
                        if 
                        bin_value_to_int(p.binval) >= bin_value_to_int(h.binval)]
    
    if not left_predecessors:
        return None

    # Se ci sono solo due predecessori, prendi quello con valore binario più piccolo
    if len(left_predecessors) == 2:
        # Ordina per valore binario crescente e prendi il primo (più piccolo)
        sorted_preds = sorted(left_predecessors, 
                            key=lambda x: bin_value_to_int(x.binval))
        rightmost = sorted_preds[0]
    else:
        rightmost = min(left_predecessors, key=lambda x: bin_value_to_int(x.binval))
    
    rightmost = set_fields(rightmost, M)
    return rightmost

def global_initial(h: Hypothesis, M) -> Hypothesis:
    n_bits = M.shape[1]
    bin_list = list(h.binval)

    # Step 1: Complementa il primo '0' che trovi da sinistra
    found_zero = False
    for i in range(len(bin_list)):
        if bin_list[i] == '0':
            bin_list[i] = '1'
            found_zero = True
            break
    if not found_zero:
        print("Nessuno zero trovato in binval, h =", h.binval)
        return h

    # Step 2: Complementa il primo '1' che trovi da destra
    found_one = False
    for i in range(len(bin_list) - 1, -1, -1):
        if bin_list[i] == '1':
            bin_list[i] = '0'
            found_one = True
            break
    if not found_one:
        print("Nessun uno trovato in binval dopo il primo zero, h =", h.binval)
        return h

    new_binval = ''.join(bin_list)
    h_new = Hypothesis(idx=None, binval=new_binval, n_bits=n_bits)
    h_new = set_fields(h_new, M)
    print(h_new.binval)
    return h_new

def remove_hr_from_current(current, h_second):
    ref_value = bin_value_to_int(h_second.binval)
    # Mantieni solo le ipotesi con valore binario minore o uguale a ref_value
    return [hr for hr in current if bin_value_to_int(hr.binval) <= ref_value]

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

matrix = preprocess_matrix_to_columns("74L85.001.matrix")
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
            next_list.extend(children)
        elif LM1(h.binval) != '0':
            h_second = global_initial(h, matrix)
            current = remove_hr_from_current(current, h_second)
            if current:  # Verifica che current non sia vuoto
                h_prime = current[0]
                if h_prime != h:
                    children = generate_children(h, current, matrix)
                    merge(next_list, children)
                    
    current = next_list

print(f"Trovate {len(soluzioni)} soluzioni:")
for i, h in enumerate(soluzioni):
    print(f"Soluzione {i+1}: binval = {h.binval}, vector = {h.vector}")