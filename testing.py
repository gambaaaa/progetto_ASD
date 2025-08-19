import numpy as np
from typing import Optional
import time
import os
from tqdm import tqdm

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
    Legge la matrice per righe dal file, rimuove le colonne che contengono solo zeri,
    e restituisce:
      - la matrice trasposta (per colonne)
      - la dimensione originale (righe, colonne)
      - la dimensione dopo il preprocessing (righe, colonne)
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
    original_shape = matrix.shape  # dimensioni prima del preprocessing

    # Trova le colonne che NON sono tutte zero
    nonzero_cols = np.any(matrix != 0, axis=0)
    matrix = matrix[:, nonzero_cols]
    processed_shape = matrix.shape  # dimensioni dopo il preprocessing

    return matrix.T, original_shape, processed_shape

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

    return successors

def find_predecessors(h0: Hypothesis, M):
    if h0.binval == '0' * M:
        return None
    else:
        predecessors = complement_one_vectors(h0)

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

    if np.array_equal(h.vector, np.zeros(matrix.shape[0], dtype=int)):
        for i in range(matrix.shape[1]):
            h_prime_bin = ['0'] * matrix.shape[1]
            h_prime_bin[i] = '1'
            h_prime_binval = ''.join(h_prime_bin)
            h_prime = Hypothesis(idx=None, binval=h_prime_binval, n_bits=matrix.shape[1])
            set_fields(h_prime, matrix)
            children.append(h_prime)
        return children

    if not current:  # Verifica che current non sia vuoto
        return children
    
    print("h = ", h.binval)
    for i in range(0, LM1(h.binval)):
        # Crea h_prime
        current_h_first = current[i]
        h_prime_bin_list = [int(b) for b in h.binval]
        h_prime_bin_list[i] = 1
        h_prime = Hypothesis(i, binval=bin_value_from_array(h_prime_bin_list), n_bits=matrix.shape[1])

        set_fields(h_prime, matrix)
        h_prime.vector = propagate(h, h_prime)

        h_initial = initial(h, h_prime, matrix)
        h_final = final(h, h_prime, matrix)

        cont = 0
        # Per ogni iterazione, parti dal primo elemento di current
        while (current_h_first is not None and
            bin_value_to_int(current_h_first.binval) <= bin_value_to_int(h_initial.binval) and
            bin_value_to_int(current_h_first.binval) >= bin_value_to_int(h_final.binval)):
            
            if (dist(bin_value(current_h_first), bin_value(h_prime)) == 1 and
                dist(bin_value(current_h_first), bin_value(h)) == 2):
                current_h_first.vector = propagate(current_h_first, h_prime)
                cont += 1
            current_h_first = prox(current_h_first, current)
          
        if cont == card(h):
            print("aggiunto h_prime:", h_prime.binval)
            children.append(h_prime)

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
        leftmost = sorted_preds[0]
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
    return h_new

def remove_hr_from_current(current, h_second):
    ref_value = bin_value_to_int(h_second.binval)
    # Mantieni solo le ipotesi con valore binario strettamente minore di ref_value
    for hr in current[:]:
        if bin_value_to_int(hr.binval) > ref_value:
            current.remove(hr)
    return current

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
        
# Ciclo su tutti i file .matrix nella cartella
# Nome della cartella di input
folder_name = "benchmarks1"

timeout_sec = 2  # Durata massima in secondi 
ask_every_level = True  # Chiedo all'utente ad ogni livello se continuare

matrix_files = [os.path.join(folder_name, f) for f in os.listdir(folder_name) if f.endswith(".matrix")]
start_time = time.time()

for input_filename in matrix_files:
    print(f"\n--- Processing file: {input_filename} ---")
    matrix, orig_shape, proc_shape = preprocess_matrix_to_columns(input_filename)
    n_bits = matrix.shape[1]
    h0 = Hypothesis(0, bin_value_from_array(np.zeros(n_bits, dtype=int)), n_bits=n_bits)
    h0 = set_fields(h0, matrix)
    current = create_currents(h0, n_bits)
    initial_len_current = len(current)
    soluzioni = []
    interrupted = False
    interrupted_level = None

    level = 0

    while current:
        next_list = []
        level += 1
        print(f"\nLivello {level} - Numero di ipotesi correnti: {len(current)} - Soluzioni trovate: {len(soluzioni)}")
        for h in current[:]:
            if check(h):
                soluzioni.append(h)
                current.remove(h)
            elif h.binval == '0' * n_bits:
                children = generate_children(h, current, matrix)
                next_list.extend(children)
            elif LM1(h.binval) != '0':
                h_second = global_initial(h, matrix)
                current = remove_hr_from_current(current, h_second)
                if current:
                    h_prime = current[0]
                    if h_prime != h:
                        children = generate_children(h, current, matrix)
                        merge(next_list, children)
        current = next_list

        # Interruzione automatica per timeout
        if time.time() - start_time > timeout_sec:
            print(f"Timeout raggiunto ({timeout_sec} secondi).")
            interrupted = True
            interrupted_level = level
            break

    # Generazione del file di output .mhs
    output_filename = input_filename.replace(".matrix", ".mhs")

    # Calcolo cardinalità minima e massima tra i MHS trovati
    if soluzioni:
        cardinalita_mhs = [card(mhs) for mhs in soluzioni]
        min_card = min(cardinalita_mhs)
        max_card = max(cardinalita_mhs)
    else:
        min_card = max_card = 0

    summary = f""";;; Input matrix: {input_filename}
;;; Matrice di dimensioni: {orig_shape[0]} x {orig_shape[1]}
;;; Dimensione della matrice dopo il preprocessing: {proc_shape[0]} x {proc_shape[1]}
;;; Numero di ipotesi iniziali: {initial_len_current}
;;; Numero di MHS trovati: {len(soluzioni)}
;;; Cardinalità minima MHS: {min_card}
;;; Cardinalità massima MHS: {max_card}
"""
    
    summary += ";;; Lista dei Minimal Hitting Sets trovati:\n"
    for idx, mhs in enumerate(soluzioni, start=1):
        summary += f";;; \t Soluzione {mhs.binval}: {card(mhs)} bit(s)\n"

    if interrupted:
        print("Processo interrotto per timeout.")
        summary = f""";;; Input matrix: {input_filename}
;;; Matrice di dimensioni: {orig_shape[0]} x {orig_shape[1]}
;;; Dimensione della matrice dopo il preprocessing: {proc_shape[0]} x {proc_shape[1]}
;;; Numero di ipotesi iniziali: {initial_len_current}
;;; Numero di MHS trovati: {len(soluzioni)}
;;; Cardinalità minima MHS: {min_card}
;;; Cardinalità massima MHS: {max_card}
"""

        summary += (
                    f";;; Calcolo NON COMPLETATO!\n"
                    f";;; Esplorazione interrotta al livello di cardinalità {interrupted_level} (non necessariamente completato)\n"
                )
        
        summary += ";;; Lista dei Minimal Hitting Sets trovati:\n"
        
        interrupted_level = level
        with open(output_filename, "w") as f:
            f.write(summary)
            if soluzioni:
                    for i in range(len(soluzioni[0].binval)):
                        row_bits = [mhs.binval[i] for mhs in soluzioni]
                        f.write(" ".join(row_bits) + " -\n")
                        
        with open("error.log", "w") as f:
            f.write(summary)
            if soluzioni:
                for i in range(len(soluzioni[0].binval)):
                    row_bits = [mhs.binval[i] for mhs in soluzioni]
                    f.write(" ".join(row_bits) + " -\n")
        break         
    else:
        summary += ";;; Calcolo COMPLETATO\n"

    with open(output_filename, 'w') as f:
        f.write(summary)
        if soluzioni:
            for i in range(len(soluzioni[0].binval)):
                row_bits = [mhs.binval[i] for mhs in soluzioni]
                f.write(" ".join(row_bits) + " -\n")
    print(f"\nMHS trovati: {len(soluzioni)}")
    print(f"Risultati salvati nel file: {output_filename}")