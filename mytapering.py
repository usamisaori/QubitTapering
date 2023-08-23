#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, NewType


# In[2]:


npmatrix = NewType('npmatrix', np.matrix)
nparray = NewType('nparray', np.ndarray)


# # 0. Tool functions

# In[3]:


from functools import reduce

# eg. Dag(|a>) = <a|
Dag = lambda matrix: matrix.conj().T
# eg. Kron(I, X, Y) = I ⊗ X ⊗ Y，计算张量用
Kron = lambda *matrices: reduce(np.kron, matrices)


# In[4]:


I = np.eye(2)

# pauli matrixes
X = np.matrix([
    [0, 1], [1, 0]
])
Y = np.matrix([
    [0, -1j], [1j, 0]
])
Z = np.matrix([
    [1, 0], [0, -1]
])


# # 1. PauliWords DataStructure

# In[5]:


# PauliWords: 1.0 XX + 1.0 XY + 1.0 XI
# PauliWord: 1.0 XX
# PauliOp: X(qubit=1)


# ## 1.1 PauliOp

# In[7]:


class PauliOp:
    def __init__(self, op_type: str, index: int):
        if op_type not in ["I", "X", "Y", "Z"]:
            raise ValueError(f"operator tpye: {op_type} is not allowed!")
        self.type = op_type # I, X, Y, Z
        self.index = index
        
    @property
    def matrix(self) -> npmatrix:
        if self.type == "I":
            return I
        elif self.type == "X":
            return X
        elif self.type == "Y":
            return Y
        elif self.type == "Z":
            return Z
        
    def __str__(self) -> str:
        return f"{self.type} (qubit={self.index})"
    
    def __repr__(self) -> str:
        return f"{self.type} (qubit={self.index})"


# ## 1.2 PauliWord

# In[8]:


class PauliWord:
    def __init__(self, op_type_str: str, coeff: complex = 1.0):
        self.num_qubits = len(op_type_str)
        self.op_type_str = op_type_str
        self.ops = []
        for idx, op_type in enumerate(op_type_str):
            self.ops.append(PauliOp(op_type, idx))
        self.coeff = coeff
        
    @property
    def matrix(self) -> npmatrix:
        ops = []
        for op in self.ops:
            ops.append(op.matrix)
        
        return self.coeff * Kron(*ops)
    
    def eliminate(self, eliminate_qubits_indexes: Union[int, List[int]]) -> None:
        if not isinstance(eliminate_qubits_indexes, list):
            eliminate_qubits_indexes = [eliminate_qubits_indexes]
        
        self.num_qubits -= len(eliminate_qubits_indexes)
        op_type_str = self.op_type_str
        op_type_arr = list(op_type_str)
        for index in eliminate_qubits_indexes:
            op_type_arr[index] = ""
        self.op_type_str = "".join(op_type_arr)
        
        ops = []
        for i, op in enumerate(self.ops):
            if i not in eliminate_qubits_indexes:
                ops.append(op)
        self.ops = ops
    
    def __mul__(self, other: PauliWord) -> PauliWord:
        if len(self.ops) != len(other.ops):
            raise ValueError("Different size PauliWord cannot be multiplied")
            
        coeff = self.coeff * other.coeff
        return_op_type_arr = []
        for op_l, op_r in zip(self.ops, other.ops):
            if op_l.type == op_r.type: # XX = I, YY = I, ZZ = I
                return_op_type_arr.append("I")
            elif op_l.type == "I": # IX = X
                return_op_type_arr.append(op_r.type)
            elif op_r.type == "I": # XI = X
                return_op_type_arr.append(op_l.type)
            elif op_l.type == "X" and op_r.type == "Y": # XY = iZ
                coeff = 1j * coeff
                return_op_type_arr.append("Z")
            elif op_l.type == "Y" and op_r.type == "X": # YX = -iZ
                coeff = -1j * coeff
                return_op_type_arr.append("Z")
            elif op_l.type == "X" and op_r.type == "Z": # XZ = -iY
                coeff = -1j * coeff
                return_op_type_arr.append("Y")
            elif op_l.type == "Z" and op_r.type == "X": # ZX = iY
                coeff = 1j * coeff
                return_op_type_arr.append("Y")
            elif op_l.type == "Y" and op_r.type == "Z": # YZ = iX
                coeff = 1j * coeff
                return_op_type_arr.append("X")
            elif op_l.type == "Z" and op_r.type == "Y": # ZY = -iX
                coeff = -1j * coeff
                return_op_type_arr.append("X")
                
        return PauliWord(''.join(return_op_type_arr), coeff)
        
    def __str__(self) -> str:
        pauli_word = [ op.type for op in self.ops ]
        return f"{self.coeff:.8f} {''.join(pauli_word)}"
    
    def __repr__(self) -> str:
        pauli_word = [ op.type for op in self.ops ]
        return f"{self.coeff:.8f} {''.join(pauli_word)}"


# ## 1.3 PauliWords

# In[9]:


from collections import defaultdict

class PauliWords:
    def __init__(self, op_type_strs: List[str], coeffs: Optional[List[complex]] = None):
        if coeffs != None and len(op_type_strs) != len(coeffs):
            raise ValueError("size of coeffs and size of op_type_strs should be the same!")
        if len(op_type_strs) == 0:
            raise ValueError("op_type_strs shouldn't be empty!")
        
        self.num_terms = len(op_type_strs)
        self.num_qubits = len(op_type_strs[0])
        if coeffs == None:
            coeffs = [1.0] * self.num_terms
        self.terms = []
        for coeff, op_type_str in zip(coeffs, op_type_strs):
            self.terms.append(PauliWord(op_type_str, coeff))
            
    @property
    def matrix(self) -> npmatrix:
        sub_hamis = [ term.matrix for term in self.terms ]
        return sum(sub_hamis)
    
    def simplify(self) -> PauliWords:
        term_dict = defaultdict(complex)
        for term in self.terms:
            term_dict[term.op_type_str] += term.coeff
            
        terms = []
        coeffs = []
        term_dict = { k:v for k,v in term_dict.items() if not abs(v) < 1e-10 }
        for k, v in term_dict.items():
            terms.append(PauliWord(k, v))
            coeffs.append(v)
            
        self.terms = terms
        self.coeffs = coeffs
        self.num_terms = len(terms)
            
        return self
    
    def eliminate(self, eliminate_qubits_indexes: Union[int, List[int]]) -> None:
        if not isinstance(eliminate_qubits_indexes, list):
            eliminate_qubits_indexes = [eliminate_qubits_indexes]
        
        self.num_qubits -= len(eliminate_qubits_indexes)
        for term in self.terms:
            term.eliminate(eliminate_qubits_indexes)
    
    def __mul__(self, other: PauliWords) -> PauliWords:
        terms = []
        for term_l in self.terms:
            for term_r in other.terms:
                terms.append(term_l * term_r)
        
        op_type_strs = []
        coeffs = []
        for term in terms:
            op_type_strs.append(term.op_type_str)
            coeffs.append(term.coeff)
            
        return PauliWords(op_type_strs, coeffs)
            
    def __str__(self) -> str:
        returns = []
        for pauliword in self.terms:
            returns.append(str(pauliword))
        return "\n".join(returns)
            
    def __repr__(self) -> str:
        returns = []
        for pauliword in self.terms:
            returns.append(repr(pauliword))
        return "\n".join(returns)


# # 2. Construct Binary Matrix G(Gx | Gz) and parity check matrix E

# In[10]:


# https://arxiv.org/pdf/1701.08213.pdf


# In[11]:


def print_G(G_x: nparray, G_z: nparray) -> None:
    row = len(G_x)
    col = len(G_x[0])
    G_str = ''
    
    for r in range(row):
        for c in range(col):
            G_str += f"  {int(G_x[r][c])}"
        G_str += '\n'
    G_str += ' ' + '-' * 3 * col + '\n'
    
    for r in range(row):
        for c in range(col):
            G_str += f"  {int(G_z[r][c])}"
        G_str += '\n'
        
    print(G_str)


# In[12]:


def print_E(E: nparray) -> None:
    row = len(E)
    col = len(E[0])
    E_str = ''
    
    for r in range(row):
        for c in range(col):
            if c == col // 2:
                E_str += "  |"
            E_str += f"  {int(E[r][c])}"
        E_str += '\n'
        
    print(E_str)


# ## 2.1 Binary Matrix G(Gx, Gz)

# In[15]:


def create_binary_matrix_G(pauli_words: PauliWords) -> Tuple[nparray, nparray]:
    if not isinstance(pauli_words, PauliWords):
        raise ValueError("input should be a PauliWords instance")
    
    # size of Gx / Gz is (num_qubits, num_terms)
    num_terms = pauli_words.num_terms
    num_qubits = pauli_words.num_qubits
    G_x = np.zeros((num_qubits, num_terms))
    G_z = np.zeros((num_qubits, num_terms))
    
    for col_idx, term in enumerate(pauli_words.terms):
        for row_idx, op in enumerate(term.ops):
            if op.type == 'X':
                G_x[row_idx][col_idx] = 1
            elif op.type == 'Y':
                G_x[row_idx][col_idx] = 1
                G_z[row_idx][col_idx] = 1
            elif op.type == 'Z':
                G_z[row_idx][col_idx] = 1
    
    return G_x, G_z


# ## 2.2 parity check matrix

# In[16]:


def create_parity_check_matrix_E(G_x: nparray, G_z: nparray) -> nparray:
    E_x = G_z.T
    E_z = G_x.T
    
    return np.hstack((E_x, E_z))


# # 3. kernel calculation and create Generators

# In[17]:


# E --(need: Gauss Jordan elimination)--> kernel(E)
#   --> generators --> paulix_ops


# ## 3.0 Gauss Jordan elimination

# In[18]:


def remove_zeros_rows(m: nparray) -> List[nparray]:
    return_m = []
    for row in m:
        if sum(row) > 0:
            return_m.append(row)
    
    return return_m


# In[19]:


def xor(m: nparray, i: int, j: int) -> nparray:
    for k in range(len(m[0])):
        m[j][k] ^= m[i][k]
    
    return m


# In[20]:


def perform_gauss_jordan_elimination(m: List[nparray]) -> nparray:
    dimension = len(m)
    
    # 1. Forward Elimination
    r = 0
    right_most_col = 0
    lowest_row = 0
    
    for c in range(len(m[0]) - 1):
        _swap = False
        _xor  = False
        for j in range(r + 1, dimension):
            if m[r][c] == 0 and m[j][c] == 1:
                m[r], m[j] = m[j], m[r]
                _swap = True
            if m[r][c] == 1:
                _xor = True
                if m[j][c] == 1:
                    m = xor(m, r, j)

        if m[r][c] == 1:
            right_most_col = c
            lowest_row = r
        if _swap or _xor:
            r += 1

    # 2. Backward Substitution
    r = lowest_row
    for c in range(right_most_col, 0, -1):
        _xor = False
        
        for j in range(r - 1, -1, -1):
            if m[r][c] == 1 and m[j][c] == 1:
                _xor = True
                m = xor(m, r, j)
                    
        if m[r][c - 1] == 0:
            r -= 1

    return m


# In[21]:


def solve_GJE(m: nparray) -> nparray:
    if len(m[0]) > 2:
        m = remove_zeros_rows(m)
        m = perform_gauss_jordan_elimination(m)
        
    return m


# ## 3.1 calculate kernel of E

# In[24]:


from sympy import Matrix

def kernel_of_E(E: nparray) -> nparray:
    E_ = solve_GJE(E.astype(int))
    E_ = Matrix(E_)
    kernel_vectors = []
    
    for vector in E_.nullspace():
        kernel_vectors.append(
            abs(np.array(vector)) 
                .flatten()
                .astype(int)
        ) # => eg. [1, 0, 0, 1]
    
    return np.array(kernel_vectors)


# ## 3.2 kernel => generators

# In[25]:


def get_generator_from_kernel(kernel_vector: nparray) -> PauliWord:
    num_qubits = len(kernel_vector) // 2
    
    op_type_arr = []
    for qubit_idx in range(num_qubits):
        if kernel_vector[qubit_idx] == 1 and kernel_vector[qubit_idx + num_qubits] == 1:
            op_type_arr.append("Y")
        elif kernel_vector[qubit_idx] == 1:
            op_type_arr.append("X")
        elif kernel_vector[qubit_idx + num_qubits] == 1:
            op_type_arr.append("Z")
        else:
            op_type_arr.append("I")
    op_type_str = ''.join(op_type_arr)
    
    return PauliWord(op_type_str)


# In[26]:


def get_generators_from_kernel(kernel_vectors: nparray) -> PauliWords:
    if len(kernel_vectors) == 0:
        raise ValueError("input kernel is empty!")
        
    generators = [ get_generator_from_kernel(kernel_vector).op_type_str 
                  for kernel_vector in kernel_vectors ]
    
    return PauliWords(generators)


# ## 3.3 generators => corresponding paulixop

# ### - tool functions to check commutes or anti-commutes property

# In[27]:


def is_commutes(generator: PauliWord, paulix_index: int) -> bool:
    # paulix_op: III...X..II
    # generator: ???...△..II
    # only need to check X△ = △X => △ = 'X' or 'I'
    return generator.ops[paulix_index].type == 'X' or \
        generator.ops[paulix_index].type == 'I'


# In[28]:


def is_anti_commutes(generator: PauliWord, paulix_index: int) -> bool:
    # paulix_op: III...X..II
    # generator: ???...△..II
    # only need to check X△ = -△X => △ = 'Z' or 'Y'
    return generator.ops[paulix_index].type == 'Z' or \
        generator.ops[paulix_index].type == 'Y'


# ### - get paulix_op from generator

# In[29]:


def get_paulix_op_from_generator(i: int, generator: PauliWord, generators: PauliWords) -> Tuple[bool, Optional[PauliWord]]:
    for op in reversed(generator.ops):
        if op.type != 'I':
            op_type_arr = ["I"] * generator.num_qubits
            op_type_arr[op.index] = "X"
            paulix_op = PauliWord(''.join(op_type_arr))
            
            # X_q(i) anti-commutes with tau_i
            if not is_anti_commutes(generator, op.index): continue
            # X_q(i) commutes with tau_j (j≠i)
            for j, generator_j in enumerate(generators.terms):
                if j != i:
                    if not is_commutes(generator_j, op.index): continue
            return True, paulix_op
    
    return False, None


# In[30]:


def get_paulix_ops_from_generators(generators: PauliWords) -> Tuple[PauliWords, PauliWords]:
    rechecked_generators = []
    paulix_ops = []
    
    for i, generator in enumerate(generators.terms):
        flag, paulix_op = get_paulix_op_from_generator(i, generator, generators)
        if flag:
            rechecked_generators.append(generator.op_type_str)
            paulix_ops.append(paulix_op.op_type_str)
    
    return PauliWords(rechecked_generators), PauliWords(paulix_ops)


# # 4. Construct unitary matrix U

# In[31]:


# generators + paulix_ops    optimal sector
#            ↘                   ↙
#       unitary matrix U     adjusted Hamilton
#              ↘             ↙
#                  H' = U†HU


# In[32]:


def construct_U(generators: PauliWords, paulix_ops: PauliWords) -> List[PauliWords]:
    c = 1 / (2 ** 0.5) 
    Us = []
    
    for generator, paulix_op in zip(generators.terms, paulix_ops.terms):
        Us.append(PauliWords(
            [paulix_op.op_type_str, generator.op_type_str],
            [c * paulix_op.coeff, c * generator.coeff]
        ))
        
    return Us


# ## 4.1 Adjust Hamilton by optimal sector

# In[33]:


# eg. H: XYYX, paulix_ops: IXII IIXI IIIX, sectors: [1, -1, -1]
# XYYX --(applying: IXII IIXI)--> [1, -1] -> -XYYX


# ### - get optimal sector

# In[34]:


def optimal_sector(Hami: PauliWords, generators: PauliWords, active_electrons: int) -> List[int]:
    num_orbitals = Hami.num_qubits

    if active_electrons > num_orbitals:
        raise ValueError(
            f"Number of active orbitals cannot be smaller than number of active electrons;"
            f" got 'orbitals'={num_orbitals} < 'electrons'={active_electrons}."
        )

    hf_str = np.where(np.arange(num_orbitals) < active_electrons, 1, 0)

    perm = []
    for generator in generators.terms:
        symmstr = np.array([1 if generator.ops[qubit].type != 'I' else 0 for qubit in range(Hami.num_qubits)])
        coeff = -1 if np.logical_xor.reduce(np.logical_and(symmstr, hf_str)) else 1
        perm.append(coeff)

    return perm


# ### - pauli x index of paulix_ops

# In[35]:


def get_paulix_op_indexes(paulix_ops: PauliWords) -> List[int]:
    x_indexes = []
    
    for paulix_op in paulix_ops.terms:
        for op in paulix_op.ops:
            if op.type == "X":
                x_indexes.append(op.index)
                break
    
    return x_indexes


# ### - adjust hamilton by optimal sector

# In[36]:


def adjust_hamilton_by_optimal_sector(Hami: PauliWords, paulix_ops: PauliWords, optimal_sector: List[int]) -> PauliWords:
    x_indexes = get_paulix_op_indexes(paulix_ops)
    
    terms = []
    coeffs = []
    for i, term in enumerate(Hami.terms):
        terms.append(term.op_type_str)
        coeff = term.coeff
        
        for x_index, sector in zip(x_indexes, optimal_sector):
            if term.ops[x_index].type != "I" and term.ops[x_index].type != "X":
                coeff *= sector
        
        coeffs.append(coeff)
            
    return PauliWords(terms, coeffs)


# ## 4.2 calculate H' = U†HU

# In[37]:


def unitary_transform(Us: List[PauliWords], Hami: PauliWords) -> PauliWords:
    H_prime = Hami
    
    for U in Us:
        H_prime = U * H_prime * U
        H_prime = H_prime.simplify()
        
    return H_prime


# # 5. Eliminate extra qubits H' => H''

# In[38]:


def eliminate_qubits(Hami_prime: PauliWords, paulix_ops: PauliWords) -> PauliWords:
    eliminate_qubits_indexes = []
    for paulix_op in paulix_ops.terms:
        for op in paulix_op.ops:
            if op.type == 'X':
                eliminate_qubits_indexes.append(op.index)
                break
    
    Hami_prime.eliminate(eliminate_qubits_indexes)
    
    return Hami_prime.simplify()


# ## 6. Hamiltonian to PauliWords

# In[40]:


from pennylane import operation, Hamiltonian


# In[41]:


def switch_op(operator: operation) -> Tuple[npmatrix, str]:
    try:
        name = operator.base_name
    except:
        name = operator
    
    if name == "Identity" or name == "I":
        op = I; label = "I"
    elif name == "PauliX" or name == "X":
        op = X; label = "X"
    elif name == "PauliY" or name == "Y":
        op = Y; label = "Y"
    elif name == "PauliZ" or name == "Z":
        op = Z; label = "Z"
        
    return op, label


# In[42]:


def create_pauliwords_from_hamilton(H: Hamiltonian, qubits: int) -> PauliWords:
    coeffs = []
    terms = []
    for i, op in enumerate(H.ops):
        op_labels = ['I' for i in range(qubits)]
        sub_Hami = [I for i in range(qubits)]

        if type(op) == operation.Tensor:
            for ob in op.obs:
                operator, label = switch_op(ob)
                idx = int(ob.wires.labels[0])
                sub_Hami[idx] = operator
                op_labels[idx] = label
        else:
            operator, label = switch_op(op)
            idx = int(op.wires.labels[0])
            sub_Hami[idx] = operator
            op_labels[idx] = label
        
        coeff = H.coeffs[i]
        coeffs.append(coeff)
        print(f"{'-' if coeff < 0 else ' '}{abs(coeff):.12f} {''.join(op_labels)}")
        terms.append(''.join(op_labels))
    
    return PauliWords(terms, coeffs)


# In[ ]:

def tapering(Hami: PauliWords, n_electrons: int) -> PauliWords:
    # generate binary matrix G(Gx|Gz) and parity check matrix E
    G_x, G_z = create_binary_matrix_G(Hami)
    E = create_parity_check_matrix_E(G_x, G_z)
    
    # get generators and corresponding paulix_ops using kernel(E)
    kernel_vectors = kernel_of_E(E)
    generators = get_generators_from_kernel(kernel_vectors)
    rechecked_generators, paulix_ops = get_paulix_ops_from_generators(generators)
    
    # get optimal sector and adjust Hamilton
    sector = optimal_sector(Hami, rechecked_generators, n_electrons)
    Hami_adjusted = adjust_hamilton_by_optimal_sector(Hami, paulix_ops, sector)
    
    # construct unitary U, and get H' = U†HU
    Us = construct_U(rechecked_generators, paulix_ops)
    Hami_prime = unitary_transform(Us, Hami_adjusted)
    
    # eliminated extra qubits
    Hami_prime_eliminated = eliminate_qubits(Hami_prime, paulix_ops)
    
    return Hami_prime_eliminated