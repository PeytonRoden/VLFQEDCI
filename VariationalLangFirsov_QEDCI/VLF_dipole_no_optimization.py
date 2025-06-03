



import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
import copy
import tqdm

import torch_optimizer as optim
import torch.optim as optimizer
import os
import random
import itertools
import matplotlib.pyplot as plt
from .QEDHF import QED_HF




#I believe it should be necessary to use the dipole basis to acheive good results here

class QED_CASCI_VLF_NO_OPT:


  def __init__(self, mol_str, psi4_options_dict, options_dict ):


    #parse options dict
    self.omega = options_dict["omega"] #frequency a.u.
    self.coherent_state = options_dict["coherent_state"] #use coherent state basis or not, True for coherent state basis
    self.photon_basis_size = options_dict["photon_basis_size"] # need to have 2 photons basis states, 0 and 1 to see polaritons
    self.lambda_vector = np.array(options_dict["lambda_vector"] ) # for example [0.0,0.0,0.05], a.u.
    self.reference_type = options_dict["reference_type"] #qedhf or hf or qedhf_dipole
    self.minimization_list = options_dict["minimization_list"] #list of which eigenvalues to minimize e.g. eigenvalues zero is ground state
    self.excitation_level =  options_dict["excitation_level"] # singles, doubles, triples, quadruples, ... for fci do total num electrons
    self.num_active_electrons = options_dict["num_active_electrons"] #number electrons which are active in CAS
    self.num_active_orbitals = options_dict["num_active_orbitals"] # number of active spin orbitals, divide by two to get number of active spatial orbitals

    self.lf_params_guess = options_dict["lf_param_guess"]

    self.model = self.CASCI_VLF( mol_str, psi4_options_dict,  self.omega, self.photon_basis_size, self.lambda_vector, self.excitation_level, self.num_active_electrons, self.num_active_orbitals, coherent_state = self.coherent_state, number_to_minimize = 4, lf_params_guess= self.lf_params_guess, reference_type = self.reference_type)
      
  @staticmethod
  def spin_block_tei(I):
    '''
    Spin blocks 2-electron integrals
    Using np.kron, we project I and I tranpose into the space of the 2x2 ide
    The result is our 2-electron integral tensor in spin orbital notation
    '''
    identity = np.eye(2)
    I = np.kron(identity, I)
    return np.kron(identity, I.T)

  @staticmethod
  def ao_to_mo(hao, C):
    '''
    Transform hao, which is the core Hamiltonian in the spin orbital basis,
    into the MO basis using MO coefficients
    '''
    
    return np.einsum('pQ, pP -> PQ', 
           np.einsum('pq, qQ -> pQ', hao, C, optimize=True), C, optimize=True)
    
  @staticmethod
  def ao_to_mo_tei(gao, C):
      '''
      Transform gao, which is the spin-blocked 4d array of physicist's notation,
      antisymmetric two-electron integrals, into the MO basis using MO coefficients

      '''
      
      return np.einsum('pQRS, pP -> PQRS',
            np.einsum('pqRS, qQ -> pQRS',
            np.einsum('pqrS, rR -> pqRS', 
            np.einsum('pqrs, sS -> pqrS', gao, C, optimize=True), C, optimize=True), C, optimize=True), C, optimize=True)



  @staticmethod
  def create_creation_operator(N):
      """
      Creates the matrix representation of the creation operator (a†) for a harmonic oscillator
      in a Hilbert space with N levels.

      Parameters:
      N (int): Number of levels.

      Returns:
      np.ndarray: The matrix representation of the creation operator.
      """
      a_dagger = np.zeros((N, N), dtype=float)

      for j in range(1, N):
          a_dagger[j, j-1] = np.sqrt(j)

      return a_dagger

  @staticmethod
  def create_annihilation_operator(N):
    """
    Creates the matrix representation of the annihilatino operator (a) for a harmonic oscillator
    in a Hilbert space with N levels.

    Parameters:
    N (int): Number of levels.

    Returns:
    np.ndarray: The matrix representation of the annihilation operator.
    """
    a = np.zeros((N, N), dtype=float)

    for j in range(1, N):
        a[j-1, j] = np.sqrt(j)

    return a

  @staticmethod
  def generate_active_space_excited_determinants( active_reference, active_virtual_orbitals, inactive_reference, N):
        """
        Generate all possible determinants for N excitations from a reference determinant.

        Parameters:
        - active_reference: Tuple of two lists, (reference_alpha, reference_beta), the occupied alpha and beta spatial orbitals in the reference determinant that are active
        - inactive_reference: Tuple of two lists, (reference_alpha, reference_beta), the occupied alpha and beta spatial orbitals in the reference determinant that are inactive
        - active_virtual_orbitals: Tuple of two lists, (virtual_alpha, virtual_beta), the unoccupied alpha and beta spatial orbitals, in the active virtual space
        - N: Integer, the number of excitations (1 for single, 2 for double, etc.).

        Returns:
        - A list of tuples, where each tuple represents an excited determinant as (excited_alpha, excited_beta).
        """
        reference_alpha, reference_beta = active_reference
        virtual_alpha, virtual_beta = active_virtual_orbitals

        # Generate all combinations of N occupied and N virtual orbitals for alpha and beta spins
        excited_determinants = []

        # Loop over possible numbers of alpha and beta excitations
        for n_alpha in range(N + 1):
            n_beta = N - n_alpha

            # Skip if n_alpha or n_beta exceeds the number of available orbitals
            if n_alpha > len(reference_alpha) or n_beta > len(reference_beta):
                continue
            if n_alpha > len(virtual_alpha) or n_beta > len(virtual_beta):
                continue

            # Generate all combinations of alpha excitations
            for occ_alpha in itertools.combinations(reference_alpha, n_alpha):
                for virt_alpha in itertools.combinations(virtual_alpha, n_alpha):
                    # Generate all combinations of beta excitations
                    for occ_beta in itertools.combinations(reference_beta, n_beta):
                        for virt_beta in itertools.combinations(virtual_beta, n_beta):
                            # Construct the excited determinant
                            excited_alpha = sorted(list(set(reference_alpha) - set(occ_alpha)) + list(virt_alpha))
                            excited_beta = sorted(list(set(reference_beta) - set(occ_beta)) + list(virt_beta))
                            excited_determinants.append(( tuple(excited_alpha), tuple(excited_beta)))

        
        i = 0

        for excited_determinant in excited_determinants:
            excited_determinants[i] = tuple( (  tuple(list(inactive_reference[0] + list(excited_determinant[0]))) ,  tuple(list(inactive_reference[1] + list(excited_determinant[1])))) ) 
            i +=1


        return excited_determinants

  @staticmethod
  def generate_orbital_partitions(number_total_electrons, number_active_electrons, number_active_orbitals):
        """
        Generate inactive reference, active reference, and active virtual orbitals 
        based on the number of total electrons, active electrons, and active orbitals.

        Parameters:
        - number_total_electrons (int): The total number of electrons in the system.
        - number_active_electrons (int): The number of active electrons.
        - number_active_orbitals (int): The number of active orbitals.

        Returns:
        - inactive_reference: Tuple of two lists, (inactive_alpha, inactive_beta).
        - active_reference: Tuple of two lists, (active_alpha, active_beta).
        - active_virtual: Tuple of two lists, (virtual_alpha, virtual_beta).
        """

        # Number of doubly occupied (inactive) orbitals
        num_inactive = (number_total_electrons - number_active_electrons) // 2  
        num_active = number_active_electrons // 2  # Alpha and beta electrons in active space

        # Inactive reference orbitals (fully occupied)
        inactive_alpha = list(range(num_inactive))
        inactive_beta = list(range(num_inactive))

        # Active reference orbitals (partially occupied in active space)
        active_alpha = list(range(num_inactive, num_inactive + num_active))
        active_beta = list(range(num_inactive, num_inactive + num_active))

        # Active virtual orbitals (remaining unoccupied orbitals in active space)
        virtual_alpha = list(range(num_inactive + num_active, num_inactive + num_active + (number_active_orbitals - number_active_electrons) // 2))
        virtual_beta = list(range(num_inactive + num_active, num_inactive + num_active + (number_active_orbitals - number_active_electrons) // 2))

        return (inactive_alpha, inactive_beta), (active_alpha, active_beta), (virtual_alpha, virtual_beta)



  @staticmethod
  def compute_phase_factor(det_I, det_J):
    alphas_I, betas_I = det_I
    alphas_J, betas_J = det_J


    reference_order_alpha = sorted(list(set(alphas_I + alphas_J)))
    reference_order_beta = sorted(list(set(betas_I + betas_J)))

        # Ensure the inputs are in the right form (lists of tuples)
    alphas_I = list(alphas_I)
    alphas_J = list(alphas_J)
    betas_I = list(betas_I)
    betas_J = list(betas_J)
    swapCounter = 0
    #while True:

    swapped_alphas = []

    for i in range(len(alphas_I)):
        pos_I = i 
        alphas_I_elem = alphas_I[pos_I]

        if alphas_I_elem in alphas_J:
            index_J = alphas_J.index(alphas_I_elem)
            if index_J != pos_I:
                #swap em
                temp = alphas_J[pos_I]
                alphas_J[pos_I]  = alphas_J[index_J]
                alphas_J[index_J] = temp
                swapCounter+=1

                swapped_alphas.append(alphas_I_elem)


    remaining_alphas_I = []
    remaining_alphas_J = []

    for i in range(len(alphas_I)):
        if alphas_I[i] not in swapped_alphas:
            remaining_alphas_I.append(alphas_I[i])
    
    for i in range(len(alphas_J)):
        if alphas_J[i] not in swapped_alphas:
            remaining_alphas_J.append(alphas_J[i])

    phase_alpha = (-1) ** swapCounter

    swapped_betas = []

    swapCounter = 0

    for i in range(len(betas_I)):
        pos_I = i 
        betas_I_elem = betas_I[pos_I]

        if betas_I_elem in betas_J:
            index_J = betas_J.index(betas_I_elem)
            if index_J != pos_I:
                #swap em
                temp = betas_J[pos_I]
                betas_J[pos_I]  = betas_J[index_J]
                betas_J[index_J] = temp
                swapCounter+=1

                swapped_betas.append(betas_I_elem)

    remaining_betas_I = []
    remaining_betas_J = []

    for i in range(len(betas_I)):
        if betas_I[i] not in swapped_betas:
            remaining_betas_I.append(betas_I[i])
    
    for i in range(len(betas_J)):
        if betas_J[i] not in swapped_betas:
            remaining_betas_J.append(betas_J[i])


    phase_beta = (-1) ** swapCounter

        #count how many are not in place in reference order for remaining beta_J
    reference_order_remaining_alpha = [x for x in reference_order_alpha if x in remaining_alphas_J]

    swapCounter = 0

    for i in range(len(reference_order_remaining_alpha)):
        pos_I = i 
        alphas_I_elem = reference_order_remaining_alpha[pos_I]

        if alphas_I_elem in remaining_alphas_J:
            index_J = remaining_alphas_J.index(alphas_I_elem)
            
            if index_J != pos_I:
                #swap em
                temp = remaining_alphas_J[pos_I]
                remaining_alphas_J[pos_I] =remaining_alphas_J[index_J]
                remaining_alphas_J[index_J] = temp
                swapCounter+=1
    phase_alpha_2 = (-1)**swapCounter


    #count how many are not in place in reference order for remaining beta_J
    reference_order_remaining_beta = [x for x in reference_order_beta if x in remaining_betas_J]

    swapCounter = 0

    for i in range(len(reference_order_remaining_beta)):
        pos_I = i 
        betas_I_elem = reference_order_remaining_beta[pos_I]

        if betas_I_elem in remaining_betas_J:
            index_J = remaining_betas_J.index(betas_I_elem)
            
            if index_J != pos_I:
                #swap em
                temp = remaining_betas_J[pos_I]
                remaining_betas_J[pos_I] =remaining_betas_J[index_J]
                remaining_betas_J[index_J] = temp
                swapCounter+=1
    phase_beta_2 = (-1)**swapCounter
    return phase_alpha * phase_beta * phase_beta_2 * phase_alpha_2



  @staticmethod 
  def compute_diff(det1, det2):


    alpha1, beta1 = det1
    alpha2, beta2 = det2


    # Find the differences between alpha and beta orbitals separately
    diff_alpha = list(set(alpha1).symmetric_difference(set(alpha2)))
    diff_beta = list(set(beta1).symmetric_difference(set(beta2)))

    # Total number of differences
    total_diff = len(diff_alpha) + len(diff_beta)

    return total_diff


  def get_energy( self ):
    energies = self.model()
    return energies


  def eval(self):
        with torch.no_grad():
            vals, vecs, H = self.model.PCQED_Hamiltonian_transformed()

        return vals, vecs, H


  class CASCI_VLF(nn.Module):


        def __init__(self, mol_str, psi4_options_dict,  omega, photon_basis_size, lambda_vector, excitation_level, num_active_electrons, num_active_orbitals, coherent_state = False, number_to_minimize = 4, lf_params_guess = None, reference_type = "qedhf"):

            super().__init__()
            self.number_to_minimize = number_to_minimize


            self.qedhf = QED_HF(mol_str, psi4_options_dict)

            self.qedhf.qed_hf(lambda_vector)

            #I in chemist notation here but we convert it to physicist and antisymmetrtize
            self.ndocc = self.qedhf.ndocc

            
            if reference_type == 'hf':
                #regular C
                self.C = self.qedhf.C_reg_HF
                self.C_ao = self.qedhf.C_reg_HF
            elif reference_type == "qedhf_dipole":
                self.C = self.qedhf.C_dipole
                self.C_ao= self.qedhf.C_dipole
            else:
                self.C = self.qedhf.C
                self.C_ao= self.qedhf.C


            self.I_ao = self.qedhf.I
            self.h_ao = self.qedhf.H_0
            self.d_ao = self.qedhf.d_ao
            self.q_ao = self.qedhf.q_ao
            self.d_n = self.qedhf.d_n
            self.d_two_body = self.qedhf.d_ao_two_body

            self.d_exp = self.qedhf.d_exp

            print("d_exp: ", self.d_exp)
            print("d_n ", self.d_n)

            self.omega = omega
            self.photon_basis_size = photon_basis_size


            

            self.num_orbs = self.C.shape[0]
            print("norbs: " ,self.num_orbs)



            self.d_ao = torch.tensor( self.d_ao, dtype= torch.double, requires_grad=False)
            self.I_ao = torch.tensor(self.I_ao, dtype= torch.double, requires_grad=False)
            self.h_ao = torch.tensor(self.h_ao, dtype= torch.double, requires_grad=False)
            #self.d_ao = torch.tensor(self.d_ao, dtype= torch.double, requires_grad=False)
            self.q_ao = torch.tensor(self.q_ao, dtype= torch.double, requires_grad=False)
            self.d_two_body_ao = torch.tensor(self.d_two_body , dtype= torch.double, requires_grad=False)


            Ca = self.C
            Cb = self.C

            self.C_ao = torch.tensor(self.C_ao, dtype= torch.double, requires_grad=False)
            self.C_ao_non_spin_blocked = torch.tensor(self.C_ao, dtype= torch.double, requires_grad=False)
            self.C = np.block([
                        [      Ca,         np.zeros_like(Cb)],
                        [np.zeros_like(Ca),          Cb     ]])
            self.C = torch.tensor(self.C, dtype= torch.double, requires_grad=False)


            norbs = self.C_ao.shape[0]
            ndocc = self.ndocc
            nvir = norbs - ndocc
            print("ndocc: ", ndocc)


            inactive_ref, active_ref, active_virtual = QED_CASCI_VLF_NO_OPT.generate_orbital_partitions(ndocc*2, num_active_electrons, num_active_orbitals)

            print("Inactive Reference:", inactive_ref)
            print("Active Reference:", active_ref)
            print("Active Virtual:", active_virtual)
            

            #build reference
            reference = []
            for i in range(ndocc):
                reference.append((i))

            reference = (reference,reference)

            self.reference = (reference,reference)
            self.reference = tuple([tuple(reference[0]), tuple(reference[1])])

            print("reference ",reference)

            virtual = []
            for i in range(ndocc, norbs):
                virtual.append(i)
            virtual = (virtual,virtual)

            basis = [tuple([tuple(reference[0]), tuple(reference[1])])   ]


            #we have to make sure singles don't couple to reference
            #singles = CI.generate_excited_determinants(reference, virtual, 1)
            self.singles = QED_CASCI_VLF_NO_OPT.generate_active_space_excited_determinants(active_reference=active_ref, active_virtual_orbitals=active_virtual, inactive_reference=inactive_ref, N= 1)

            excited_determinants = []
            for i in range(1,  excitation_level+1, 1):
                excited_determinants_new = QED_CASCI_VLF_NO_OPT.generate_active_space_excited_determinants(active_reference=active_ref, active_virtual_orbitals=active_virtual, inactive_reference=inactive_ref, N = i)
                excited_determinants = excited_determinants+ excited_determinants_new 

            basis =basis+  excited_determinants 
            basis= tuple(basis)


            print("here is the basis")
            #print(basis)

            el_basis_size = len(basis)

            #append photon state onto basis

            coupled_basis = []

            for ph_num in range(self.photon_basis_size):
                for i in range(len(basis)):
                    coupled_basis.append(((basis[i]), (ph_num)))


            coupled_basis = tuple(coupled_basis)
            self.coupled_basis = tuple(coupled_basis)
            #print(coupled_basis)

            basis_size = len(coupled_basis)
            self.basis_size = len(coupled_basis)

            print("basis _size : ", basis_size)

            print(type(QED_CASCI_VLF_NO_OPT.create_creation_operator(self.photon_basis_size)))
            self.a_dag, self.a = QED_CASCI_VLF_NO_OPT.create_creation_operator(self.photon_basis_size) , QED_CASCI_VLF_NO_OPT.create_annihilation_operator(self.photon_basis_size) 


            self.identity_photon =  torch.tensor(np.eye(self.photon_basis_size), dtype= torch.double, requires_grad=False)
            self.identity_electronic = torch.tensor( np.eye(el_basis_size), dtype= torch.double, requires_grad=False)
            self.a_dag_mult_a = self.a_dag@self.a
            self.a_dag_plus_a = self.a_dag + self.a

            self.a_dag_mult_a = torch.tensor(self.a_dag_mult_a, requires_grad=False)
            self.a_dag_plus_a = torch.tensor(self.a_dag_plus_a , requires_grad=False)


            #construuting rotation matrix, starting with random guess:    
            # # Generate a random matrix
            self.A = nn.Parameter(torch.randn(self.num_orbs, self.num_orbs,  dtype=torch.double) * 0.0000000001,  requires_grad=False)



            d_ao_diag = torch.diag(self.ao_to_mo(self.d_ao, self.C_ao_non_spin_blocked))
            #self.lang_firsov_params =  torch.tensor((np.random.rand(self.num_orbs) -0.5) * 0.000000001,  dtype= torch.double , requires_grad=False)


            if reference_type == "qedhf" or reference_type == "hf":
                self.lang_firsov_params= nn.Parameter( (1/(np.sqrt(omega*2) ) )* d_ao_diag, requires_grad=False)
            else:
                print("using dipole basis")

                print("d_n")
                print("lf params: ", (1/(np.sqrt(omega*2) ) )* torch.tensor(self.qedhf.d_eigvals)/2)

                # new_d_mo = self.qedhf.C.T @ np.diag(np.ones(self.d_ao.shape[0 ]* self.d_n/(ndocc))) @self.qedhf.C 
                # vals, vecs =  np.linalg.eigh(vals, vecs)


                # print("new lf params: " ,   (1/(np.sqrt(omega*2) ) )* torch.tensor(vals)/2  )

                self.lang_firsov_params= nn.Parameter( (1/(np.sqrt(omega*2) ) )* (torch.tensor(self.qedhf.d_eigvals)    )/2, requires_grad=True)




            #gotta refigure out this part here
            # # if lf_params_guess != None:
            # try:
            #     print(self.lang_firsov_params[0]) #try printing out first element, this will break it if its none
            #     self.lang_firsov_params = nn.Parameter( torch.tensor(lf_params_guess), requires_grad=False)

            # except:
            #     print("can't use lf param guess")



        def spin_block_tei(self, I):
            '''
            Spin blocks 2-electron integrals
            Using np.kron, we project I and I tranpose into the space of the 2x2 ide
            The result is our 2-electron integral tensor in spin orbital notation
            '''
            identity = np.eye(2)
            I = np.kron(identity, I)
            return np.kron(identity, I.T)



        def spin_block_tei_torch(self, I):
            """
            Spin blocks 2-electron integrals in PyTorch
            
            Args:
                I: A PyTorch 4D tensor representing 2-electron integrals in spatial orbital basis
                
            Returns:
                A PyTorch tensor representing 2-electron integrals in spin orbital basis
            """
            # Create 2x2 identity matrix for spin
            identity = torch.eye(2)
            
            # First Kronecker product
            # In PyTorch, we need to implement the Kronecker product for higher dimensionss
            I_spin = torch.kron(identity, I)
            
            # Second Kronecker product with transpose
            # Note: For a 4D tensor, we need to define what transpose means
            I_transposed = self.transpose_4d_tensor(I_spin)
            result = torch.kron(identity, I_transposed.contiguous())
            
            return result


        def spin_block_oei_torch(self, h):
            '''
            Spin blocks 1-electron integrals
            Using np.kron, we project I and I tranpose into the space of the 2x2 ide
            The result is our 2-electron integral tensor in spin orbital notation
            '''
            identity = torch.eye(2)
            return torch.kron(identity, h)


        def transpose_4d_tensor(self, tensor):
            """
            Implements the transpose operation for a 4D tensor
            This might need to be adjusted based on the specific definition of transpose used
            in the original function
            """
            # For a 4D tensor with indices (i,j,k,l), a common transpose might be (k,l,i,j)
            return tensor.permute(3,2,1,0)

        def ao_to_mo(self, hao, C):
            '''
            Transform hao, which is the core Hamiltonian in the spin orbital basis,
            into the MO basis using MO coefficients
            '''
            
            return torch.einsum('pQ, pP -> PQ', 
                torch.einsum('pq, qQ -> pQ', hao, C), C)
            
        def ao_to_mo_tei(self, gao, C):
            '''
            Transform gao, which is the spin-blocked 4d array of physicist's notation,
            antisymmetric two-electron integrals, into the MO basis using MO coefficients

            '''
            return torch.einsum('pQRS, pP -> PQRS',
                    torch.einsum('pqRS, qQ -> pQRS',
                    torch.einsum('pqrS, rR -> pqRS', 
                    torch.einsum('pqrs, sS -> pqrS', gao, C), C), C), C)


        @staticmethod
        def laguerre_poly(n, k, z):
            """
            Vectorized computation of generalized Laguerre polynomials L_n^(k)(z).
            """
            n = n.to(dtype=torch.int64)
            k = k.to(dtype=torch.int64)
            z = z.to(dtype=torch.float64)

            L0 = torch.ones_like(z, dtype=torch.float64)
            L1 = 1 + k - z

            result = torch.where(n == 0, L0, torch.where(n == 1, L1, torch.zeros_like(z)))

            # print(max_n)

            max_n = n.max().item()
            L_prev = L0
            L_curr = L1

            for i in range(2, max_n + 1):
                L_next = ((2 * i - 1 + k - z) * L_curr - (i - 1 + k) * L_prev) / i
                L_prev, L_curr = L_curr, L_next
                result = torch.where(n == i, L_curr, result)

            return result

        @staticmethod
        def factorial(n):
            """
            Compute the factorial of n using torch.lgamma for gradient support,
            with hard-coded handling for n = 0 and n = 1.

            Parameters:
            n (Tensor): Input tensor of non-negative integers.

            Returns:
            Tensor: Factorial of n.
            """
            # Ensure n is at least float for operations
            n = n.to(dtype=torch.float64)

            # Handle special cases manually
            out = torch.exp(torch.lgamma(n + 1.0))

            # Set factorial(0) = 1 and factorial(1) = 1 manually
            special_case = (n == 0) | (n == 1)
            out = torch.where(special_case, torch.ones_like(out), out)

            return out


        @staticmethod
        def displacement_matrix_element(n, m, z):
            """
            Compute ⟨n|e^(-z(b - b†))|m⟩ for arrays of n, m, and z.

            Parameters:
            n (Tensor): Tensor of non-negative integers.
            m (Tensor): Tensor of non-negative integers.
            z (Tensor): Displacement parameter.

            Returns:
            Tensor: Matrix elements ⟨n|e^(-z(b - b†))|m⟩.
            """

            n = n.to(dtype=torch.float64)
            m = m.to(dtype=torch.float64)
            z =- z.to(dtype=torch.float64)
            
            with torch.no_grad():

                abs_diff = torch.abs(n - m).to(dtype=torch.int64)
                min_nm = torch.minimum(n, m)
                max_nm = torch.maximum(n, m)

                # Compute factorial terms
                fact_min = QED_CASCI_VLF_NO_OPT.CASCI_VLF.factorial(min_nm)
                fact_max = QED_CASCI_VLF_NO_OPT.CASCI_VLF.factorial(max_nm)
                prefactor = torch.sqrt(fact_min / fact_max)

                # Create a mask for the condition n >= m
                n_ge_m_mask = (n >= m).to(dtype=torch.float64)
            
            # Compute power term using gradient-friendly masks
            # power_term_negative = (-z) ** abs_diff
            # power_term_positive = z ** abs_diff
                
            power_term_negative = torch.sign(-z)**abs_diff * torch.abs(z)**abs_diff
            power_term_positive = torch.sign(z)**abs_diff * torch.abs(z)**abs_diff

            #epsiolon for grad stability
            eps = 1e-14
            power_term_negative = torch.exp(abs_diff * torch.log(torch.abs(z) + eps)) * torch.sign(-z)**abs_diff
            power_term_positive = torch.exp(abs_diff * torch.log(torch.abs(z) + eps)) *  torch.sign(z)**abs_diff

            power_term = n_ge_m_mask * power_term_positive + (1 - n_ge_m_mask) * power_term_negative

            # Compute exponential term
            exp_term = torch.exp(-0.5 * z ** 2)

            # Compute Laguerre polynomial
            laguerre = QED_CASCI_VLF_NO_OPT.CASCI_VLF.laguerre_poly(min_nm.to(dtype=torch.int64), abs_diff.to(dtype=torch.int64), z ** 2)

            return prefactor * power_term * exp_term * laguerre
        


        @staticmethod
        def displacement_matrix_element_b_dag_plus_b(n, m ,z):

            """
            Compute ⟨n|e^(-z(b - b†))  (b† + b)  |m⟩ for arrays of n, m, and z.

            Parameters:
            n (Tensor): Tensor of non-negative integers.
            m (Tensor): Tensor of non-negative integers.
            z (Tensor): Displacement parameter.

            Returns:
            Tensor: Matrix elements ⟨n|e^(-z(b - b†)) (b† + b) |m⟩.
            """
                        
            n = n.to(dtype=torch.float64)
            m = m.to(dtype=torch.float64)
            z = z.to(dtype=torch.float64)
            
            # Create a mask for valid m values (m >= 1)
            m_valid_mask = (m >= 1).to(dtype=torch.float64)
            
            # Term 1: √m ⟨n|e^(-z(b - b†))|m-1⟩
            # Only compute for valid m values
            m_minus_1 = torch.maximum(m - 1, torch.zeros_like(m))  # Ensure m-1 doesn't go negative
            term1 = torch.sqrt(m) * QED_CASCI_VLF_NO_OPT.CASCI_VLF.displacement_matrix_element(n, m_minus_1, z) * m_valid_mask
            
            # Term 2: √(m+1) ⟨n|e^(-z(b - b†))|m+1⟩
            term2 = torch.sqrt(m + 1) * QED_CASCI_VLF_NO_OPT.CASCI_VLF.displacement_matrix_element(n, m + 1, z)

            return term1 + term2
            


        def forward(self):
            #return only eigvals
            return self.QEDCASCI_ENERGY()[0]



        def transform_one_body_VLF( self, h, lambda_values, n_ph):
            """
            Transform the Hamiltonian by incorporating photon displacement effects.
            Stores the one-body terms as a tensor of shape (n_ph, n_ph, n_orbs, n_orbs).
            
            Parameters:
            h (Tensor): One-body electronic Hamiltonian (n_orbs x n_orbs)
            lambda_values (Tensor): Array of lambda_p values (length n_orbs)
            n_ph (int): Maximum photon number state considered

            Returns:
            Tensor: Transformed one-body integrals stored as (n_ph x n_ph x n_orbs x n_orbs)
            """
            h = self.spin_block_oei_torch(h.contiguous()).contiguous()
            n_orbs = h.shape[0]

            # Compute the displacement matrix z = lambda_q - lambda_p
            lambda_p = lambda_values.view(-1, 1)  # Shape: (n_orbs, 1)
            lambda_q = lambda_values.view(1, -1)  # Shape: (1, n_orbs)
            z = lambda_q - lambda_p  # Shape: (n_orbs, n_orbs)

            # Vectorize the computation of displacement matrix elements D_nm
            # Create tensors for photon states n and m
            n = torch.arange(n_ph, device=h.device).view(n_ph, 1, 1, 1)  # Shape: (n_ph, 1, 1, 1)
            m = torch.arange(n_ph, device=h.device).view(1, n_ph, 1, 1)  # Shape: (1, n_ph, 1, 1)
            z_expanded = z.view(1, 1, n_orbs, n_orbs)  # Shape: (1, 1, n_orbs, n_orbs)

            # Compute D_nm for all combinations using a vectorized approach
            D_nm = QED_CASCI_VLF_NO_OPT.CASCI_VLF.displacement_matrix_element(n, m, z_expanded)  # Shape: (n_ph, n_ph, n_orbs, n_orbs)

            # Multiply h with D_nm, utilizing broadcasting
            h_expanded = h.view(1, 1, n_orbs, n_orbs)  # Shape: (1, 1, n_orbs, n_orbs)
            H_transformed = h_expanded * D_nm  # Shape: (n_ph, n_ph, n_orbs, n_orbs)

            return H_transformed

        def transform_hamiltonian_with_bdagplusb( self, h, lambda_values, n_ph):
            """
            Transform the Hamiltonian by incorporating photon displacement effects and b_dagger_plus_b.
            Stores the one-body terms as a tensor of shape (n_ph, n_ph, n_orbs, n_orbs).
            
            Parameters:
            h (Tensor): One-body electronic Hamiltonian (n_orbs x n_orbs)
            lambda_values (Tensor): Array of lambda_p values (length n_orbs)
            n_ph (int): Maximum photon number state considered
            
            Returns:
            Tensor: Transformed one-body integrals stored as (n_ph x n_ph x n_orbs x n_orbs)
            """
            h = self.spin_block_oei_torch(h.contiguous()).contiguous()
            n_orbs = h.shape[0]

            # Compute the displacement matrix z = lambda_q - lambda_p
            lambda_p = lambda_values.view(-1, 1)  # Shape: (n_orbs, 1)
            lambda_q = lambda_values.view(1, -1)  # Shape: (1, n_orbs)
            z = lambda_q - lambda_p  # Shape: (n_orbs, n_orbs)

            # Vectorize the computation of displacement matrix elements D_nm
            # Create tensors for photon states n and m
            n = torch.arange(n_ph, device=h.device).view(n_ph, 1, 1, 1)  # Shape: (n_ph, 1, 1, 1)
            m = torch.arange(n_ph, device=h.device).view(1, n_ph, 1, 1)  # Shape: (1, n_ph, 1, 1)
            z_expanded = z.view(1, 1, n_orbs, n_orbs)  # Shape: (1, 1, n_orbs, n_orbs)

            # Compute D_nm for all combinations using a vectorized approach
            D_nm = QED_CASCI_VLF_NO_OPT.CASCI_VLF.displacement_matrix_element_b_dag_plus_b(n, m, z_expanded)  # Shape: (n_ph, n_ph, n_orbs, n_orbs)

            # Multiply h with D_nm, utilizing broadcasting
            h_expanded = h.view(1, 1, n_orbs, n_orbs)  # Shape: (1, 1, n_orbs, n_orbs)
            H_transformed = h_expanded * D_nm  # Shape: (n_ph, n_ph, n_orbs, n_orbs)

            return H_transformed


        def transform_one_body_VLF_d_lang_firsov(self, h, lambda_values, n_ph):
            """
            Vectorized version of Lang-Firsov displaced one-body transformation.
            
            Returns:
            Tensor of shape (n_ph, n_ph, n_orbs, n_orbs)
            """
            h = self.spin_block_oei_torch(h.contiguous()).contiguous()
            n_orbs = h.shape[0]

            # Compute z = lambda_r - lambda_p for all p, r
            lambda_p = lambda_values.view(-1, 1)  # (n_orbs, 1)
            lambda_r = lambda_values.view(1, -1)  # (1, n_orbs)
            z = lambda_r - lambda_p  # (n_orbs, n_orbs)

            # Photon indices n and m
            n = torch.arange(n_ph, device=h.device).view(n_ph, 1, 1, 1)
            m = torch.arange(n_ph, device=h.device).view(1, n_ph, 1, 1)
            z_expanded = z.view(1, 1, n_orbs, n_orbs)

            # Compute D_nm for all combinations
            D_nm = QED_CASCI_VLF_NO_OPT.CASCI_VLF.displacement_matrix_element(n, m, z_expanded)  # (n_ph, n_ph, n_orbs, n_orbs)

            # Multiply h[p, r] * lambda_r * D_nm
            h_expanded = h.view(1, 1, n_orbs, n_orbs)  # (1,1,n_orbs,n_orbs)

            print("lambda_r: ", lambda_r)
            lambda_diag = torch.diag(lambda_values)
            lambda_r_expanded = lambda_r.view(1, 1, n_orbs)  # (1,1,n_orbs,n_orbs)
            print("lambda_r_exp: ", lambda_r_expanded)

            #H_transformed = h_expanded * lambda_r_expanded * D_nm  # Broadcasting
            H_transformed = h_expanded  * D_nm 

            H_transformed = H_transformed * lambda_r_expanded

            return H_transformed



        def transform_two_body_VLF(self, V, lambda_values, n_ph):
            """
            Vectorized version of Lang-Firsov displaced two-body transformation.
            
            Returns:
            Tensor of shape (n_ph, n_ph, n_orbs, n_orbs, n_orbs, n_orbs)
            """
            # Assume V is already antisymmetrized and spin-blocked beforehand if needed
            V=  self.spin_block_tei_torch(V.contiguous() ).contiguous()
            V = V.permute(0, 2, 1, 3) - V.permute(0, 2, 3, 1) #antisymmetrize

            n_orbs = V.shape[0]

            # Compute z_2b = lambda_s + lambda_q - lambda_r - lambda_p
            lambda_p = lambda_values.view(-1, 1, 1, 1)  # (n_orbs,1,1,1)
            lambda_r = lambda_values.view(1, -1, 1, 1)  # (1,n_orbs,1,1)
            lambda_q = lambda_values.view(1, 1, -1, 1)  # (1,1,n_orbs,1)
            lambda_s = lambda_values.view(1, 1, 1, -1)  # (1,1,1,n_orbs)

            z_2b = lambda_s + lambda_q - lambda_r - lambda_p  # (n_orbs, n_orbs, n_orbs, n_orbs)

            # Photon indices
            n = torch.arange(n_ph, device=V.device).view(n_ph, 1, 1, 1, 1, 1)
            m = torch.arange(n_ph, device=V.device).view(1, n_ph, 1, 1, 1, 1)
            z_2b_expanded = z_2b.view(1, 1, n_orbs, n_orbs, n_orbs, n_orbs)

            # Compute D_nm for all combinations
            D_nm_2b = QED_CASCI_VLF_NO_OPT.CASCI_VLF.displacement_matrix_element(n, m, z_2b_expanded)  # (n_ph, n_ph, n_orbs, n_orbs, n_orbs, n_orbs)

            # Expand V for broadcasting
            V_expanded = V.view(1, 1, n_orbs, n_orbs, n_orbs, n_orbs)

            # Final transformed two-body tensor
            V_transformed = V_expanded * D_nm_2b

            return V_transformed



        def QEDCASCI_ENERGY(self):
            """
            Build the CASCI Matrix

            """
            norbs = self.num_orbs

            # Construct the anti-Hermitian matrix: K = A - A^T
            K = self.A - self.A.T
            U = torch.linalg.matrix_exp(K)

            self.C_ao_non_spin_blocked =  self.C_ao#@ U


            self.C = torch.kron(torch.eye(2), self.C_ao_non_spin_blocked)


            self.lang_firsov_param_matrix = torch.eye(self.lang_firsov_params.shape[0]) * self.lang_firsov_params
            print("lang firsov param matrix")
            self.lang_firsov_params_spinblock =  torch.cat([self.lang_firsov_params,self.lang_firsov_params])
            self.lang_firsov_param_matrix_spinblock= self.spin_block_oei_torch(self.lang_firsov_param_matrix)

            print(self.lang_firsov_param_matrix)

            #any part that has an exponential attached is transformed here
            self.d_mo_temp = self.ao_to_mo(self.d_ao, self.C_ao_non_spin_blocked)

            self.g_mo =  self.transform_two_body_VLF( self.ao_to_mo_tei(self.I_ao, self.C_ao_non_spin_blocked) , self.lang_firsov_params_spinblock, self.photon_basis_size)
            self.d_two_body =  self.transform_two_body_VLF(self.ao_to_mo_tei(self.d_two_body_ao, self.C_ao_non_spin_blocked) , self.lang_firsov_params_spinblock, self.photon_basis_size)
            self.h_mo =  self.transform_one_body_VLF(self.ao_to_mo(self.h_ao, self.C_ao_non_spin_blocked), self.lang_firsov_params_spinblock, self.photon_basis_size)  
            
            #print("d ao shape; ", self.d_ao.shape)
            self.d_mo =  self.transform_one_body_VLF( self.ao_to_mo(self.d_ao, self.C_ao_non_spin_blocked) , self.lang_firsov_params_spinblock, self.photon_basis_size)   
            self.q_mo =  self.transform_one_body_VLF( self.ao_to_mo(self.q_ao, self.C_ao_non_spin_blocked), self.lang_firsov_params_spinblock, self.photon_basis_size)  
            #self.d_mo_non_spin_blocked =  self.transform_one_body_VLF_non_spinblocked( self.ao_to_mo(self.d_ao, self.C_ao_non_spin_blocked) , self.lang_firsov_params, self.photon_basis_size, self.C_ao_non_spin_blocked)   
            self.d_lang_firsov_one_body= self.transform_one_body_VLF_d_lang_firsov( self.d_mo_temp, self.lang_firsov_params_spinblock, self.photon_basis_size)
            

            self.d_mo_b_dag_plus_b = self.transform_hamiltonian_with_bdagplusb(self.ao_to_mo(self.d_ao, self.C_ao_non_spin_blocked) , self.lang_firsov_params_spinblock, self.photon_basis_size)

            self.lang_firsov_sq = torch.einsum('pq, pq -> pq', self.lang_firsov_param_matrix_spinblock, self.lang_firsov_param_matrix_spinblock)


            self.H_PF = torch.zeros((self.basis_size, self.basis_size) , dtype= torch.double, requires_grad=False)

            counter = 0


            print("singles; " , self.singles)
            print("reference; ", self.reference)

            for i in range(self.basis_size):
                for j in range(i, self.basis_size):

                    total_val = torch.zeros(1, dtype=torch.float64, requires_grad=False)[0]

                    #pick out phootonic and electronic parts of basis
                    photon_basis_i = self.coupled_basis[i][1]
                    photon_basis_j = self.coupled_basis[j][1]
                    el_basis_i =self.coupled_basis[i][0]
                    el_basis_j = self.coupled_basis[j][0]

                    total_diff = QED_CASCI_VLF_NO_OPT.compute_diff(tuple(el_basis_i),tuple(el_basis_j))
                    photon_diff = photon_basis_i - photon_basis_j
                    #print(photon_diff)
                    phase = 1

                    #if photon_diff == 0 or photon_diff==1 or photon_diff==-1:
                    if total_diff == 0 or total_diff == 2 or total_diff == 4:
                            phase = QED_CASCI_VLF_NO_OPT.compute_phase_factor(tuple(el_basis_i),tuple(el_basis_j))

                    phase = torch.tensor(phase, dtype=torch.float64, requires_grad=False)
        
                    #singles don't couple to reference, brilluuions theorem, not true for some cases of C
                    # if (el_basis_i in self.singles and el_basis_j == self.reference) or (el_basis_j in self.singles and el_basis_i == self.reference):
                    if False:
                        print("this shulnt run")
                    #everything else
                    else:

                        #val = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                        val = 0

                        #regular electronic hamiltonian part
                        if total_diff ==0 or total_diff==2 or total_diff ==4:
                            val = val + (QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.h_mo[photon_basis_i, photon_basis_j],tuple(el_basis_i),tuple(el_basis_j), norbs) +  QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_two_body(total_diff, self.g_mo[photon_basis_i, photon_basis_j],tuple(el_basis_i),tuple(el_basis_j),norbs))
                            val= val*phase
                        else:
                            val = val + 0
                        total_val = total_val + val

                        val = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                        #dse part
                        if total_diff ==0 or total_diff==2 or total_diff ==4:
                            val = val +( QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_two_body(total_diff, self.d_two_body[photon_basis_i, photon_basis_j], tuple(el_basis_i),tuple(el_basis_j),norbs) - 1 * QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.q_mo[ photon_basis_i, photon_basis_j],tuple(el_basis_i),tuple(el_basis_j),  norbs) )
                            val= val  + 2 * self.d_n * QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.d_mo[ photon_basis_i, photon_basis_j], tuple(el_basis_i),tuple(el_basis_j), norbs) 
                            if total_diff ==0  and photon_diff == 0:
                                val = val +self.d_n **2  
                            val= val*phase
                        else:
                            val = 0
                            #val = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)

                        total_val = total_val +  0.5 * val

                        #val = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                        val = 0
                        #blc part
                        if total_diff ==0 or total_diff==2 or total_diff==4:

                            val = val +( QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.d_mo_b_dag_plus_b[photon_basis_i, photon_basis_j], tuple(el_basis_i),tuple(el_basis_j), norbs) ) #* (a_dag + a)[photon_basis_i, photon_basis_j]
                            val=val + -2*(QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_two_body_kinda(total_diff, self.d_mo[photon_basis_i, photon_basis_j], self.lang_firsov_param_matrix_spinblock, tuple(el_basis_i),tuple(el_basis_j), norbs) ) 
                            val = val + -2 * (QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.d_lang_firsov_one_body[photon_basis_i, photon_basis_j], tuple(el_basis_i),tuple(el_basis_j),norbs))

                            if total_diff ==0:
                                val = val + self.d_n * (self.a_dag_plus_a [photon_basis_i, photon_basis_j])

                            if photon_diff == 0 and total_diff == 0:
                                val = val + self.d_n * ( - 2* (QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.lang_firsov_param_matrix_spinblock, tuple(el_basis_i),tuple(el_basis_j), norbs) ))
                            val = val*phase 
                        else:
                            #val = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                            val = 0

                        total_val = total_val+  (- np.sqrt(self.omega/2)* val)

                        #cav
                        #val = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                        val = 0

                        if total_diff == 0:
                            val = val +(self.a_dag_mult_a[photon_basis_i,photon_basis_j])
                        if total_diff ==0:
                            val = val +(-  self.a_dag[photon_basis_i,photon_basis_j]  * QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.lang_firsov_param_matrix_spinblock, tuple(el_basis_i),tuple(el_basis_j), norbs)) 
                            val =val + (-  self.a[photon_basis_i,photon_basis_j]  * QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.lang_firsov_param_matrix_spinblock, tuple(el_basis_i),tuple(el_basis_j), norbs)) 
                            #val =val + (-  self.a_dag_plus_a[photon_basis_i,photon_basis_j]  * QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.lang_firsov_param_matrix_spinblock, tuple(el_basis_i),tuple(el_basis_j), norbs)) 
                            
                        if photon_diff == 0 and ( total_diff ==0 ):
                            
                            val = val +(QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_one_body(total_diff, self.lang_firsov_sq,tuple(el_basis_i),tuple(el_basis_j), norbs ))
                            val = val + (QED_CASCI_VLF_NO_OPT.CASCI_VLF.slater_condon_two_body_kinda(total_diff, self.lang_firsov_param_matrix_spinblock, self.lang_firsov_param_matrix_spinblock, tuple(el_basis_i),tuple(el_basis_j), norbs))
                            
                        val = val* phase
                        total_val =total_val +  self.omega *  val

                        if i!=j:
                            self.H_PF[i,j] = self.H_PF[i,j]+total_val
                            self.H_PF[j,i] = self.H_PF[j,i]+total_val
                        if i==j:
                            self.H_PF[i,j] = self.H_PF[i,j]+total_val

                    #print("/n")
                    counter +=1
                        
            vals, vecs  = torch.linalg.eigh(self.H_PF)

            return vals, vecs, self.H_PF


        def return_params(self):
            return [self.A, self.lang_firsov_params]


        @staticmethod
        def slater_condon_one_body(total_diff, h_mo, det1, det2, norbs):
            alpha1, beta1 = det1
            alpha2, beta2 = det2


            # Find the differences between alpha and beta orbitals separately
            diff_alpha = list(set(alpha1).symmetric_difference(set(alpha2)))
            diff_beta = list(set(beta1).symmetric_difference(set(beta2)))

            if total_diff == 0:
                # print("identical")

                one_body = torch.einsum('i->', (h_mo[alpha1, alpha1]))
                beta1_shifted = tuple([b+norbs for b in beta1])
                one_body =one_body +  torch.einsum('i->', (h_mo[beta1_shifted, beta1_shifted]))

                return one_body 

            elif total_diff == 2:

                # Find the orbitals that differ
                if len(diff_alpha) == 2:
                    k, l = diff_alpha
                elif len(diff_beta) == 2:
                    k, l = diff_beta
                    k =k +norbs
                    l = l + norbs
                else:
                    k, l = diff_alpha[0], diff_beta[0]
                    l = l+norbs
                #adding these two lines for testing
                beta1_shifted = tuple([b+norbs for b in beta1])
                beta2_shifted =tuple([b+norbs for b in beta2])
                # Ensure k is in det1 and l is in det2
                if k not in alpha1 + beta1_shifted or l not in  alpha2 + beta2_shifted:
                    k, l = l, k
            
                one_body = h_mo[k,l]

                return one_body
            else: 
                return 0.0

        @staticmethod
        def slater_condon_two_body(total_diff, g_mo, det1, det2, norbs):
            alpha1, beta1 = det1
            alpha2, beta2 = det2


            # Find the differences between alpha and beta orbitals separately
            diff_alpha = list(set(alpha1).symmetric_difference(set(alpha2)))
            diff_beta = list(set(beta1).symmetric_difference(set(beta2)))

            if total_diff == 0:

                #two_body =  torch.tensor(0.0, dtype=torch.float64, requires_grad=False)

                beta1_shifted = [b+norbs for b in beta1]
                beta2_shifted = [b+norbs for b in beta2]

                alpha1 = list(alpha1)
                alpha2 = list(alpha2)
                
                two_body = 0.5 * torch.einsum('ijij->', g_mo[alpha1][:, alpha2][:, :, alpha1][:, :, :, alpha2])
                two_body = two_body+  0.5 * torch.einsum('ijij->', g_mo[beta1_shifted][:, beta2_shifted ][:, :, beta1_shifted][:, :, :, beta2_shifted ])
                two_body = two_body+  0.5 * torch.einsum('ijij->', g_mo[alpha1][:,  beta2_shifted][:, :, alpha1][:, :, :,  beta2_shifted])
                two_body = two_body+  0.5 * torch.einsum('ijij->', g_mo[alpha2][:, beta1_shifted][:, :, alpha2][:, :, :, beta1_shifted])
                
                return two_body

            elif total_diff == 2:

                # print("single excitation ")

                # Find the orbitals that differ
                if len(diff_alpha) == 2:
                    k, l = diff_alpha
                elif len(diff_beta) == 2:
                    k, l = diff_beta
                    k =k +norbs
                    l = l + norbs
                else:
                    k, l = diff_alpha[0], diff_beta[0]
                    l = l+norbs


                #adding these two lines for testing
                beta1_shifted = tuple([b+norbs for b in beta1])
                beta2_shifted =tuple([b+norbs for b in beta2])
                # Ensure k is in det1 and l is in det2, had to modify this also
                if k not in alpha1 + beta1_shifted or l not in  alpha2 + beta2_shifted:
                    k, l = l, k
                    # print(f"swapped {k}, {l}")

                two_body =  torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                

                alpha1 = list(alpha1)
                alpha2 = list(alpha2)
                beta1_shifted = [b+norbs for b in beta1]
                beta2_shifted =[b+norbs for b in beta2]

                # First create a mask for i != k
                alpha1_filtered = [i for i in alpha1 if i != k]
                # Then use einsum
                two_body = torch.einsum('i->', g_mo[k, alpha1_filtered, l, alpha1_filtered])
                # First create a mask for i != k
                beta1_filtered = [i for i in beta1_shifted if i != k]
                # Then use einsum
                two_body = two_body + torch.einsum('i->', g_mo[k, beta1_filtered, l, beta1_filtered])
                
                return two_body

            elif total_diff == 4:

                #print("double excitation")
                removed_alpha = list(set(alpha1) - set(alpha2))
                removed_beta = list(set(beta1) - set(beta2))
                added_alpha = list(set(alpha2) - set(alpha1))
                added_beta = list(set(beta2) - set(beta1))


                if len(removed_alpha) == 2 and len(added_alpha) == 2:  # Alpha-Alpha double excitation
                    k, m = sorted(removed_alpha)
                    l, n = sorted(added_alpha)
                    return  g_mo[k,m,l,n]

                elif len(removed_beta) == 2 and len(added_beta) == 2:  # Beta-Beta double excitation
                    k, m = sorted(removed_beta)
                    l, n = sorted(added_beta)
                    k = k+norbs
                    m = m+norbs
                    l = l+norbs
                    n = n+norbs
                    return  g_mo[k,m,l,n]
                
                elif len(removed_alpha) == 1 and len(removed_beta) == 1 and len(added_alpha) == 1 and len(added_beta) == 1:  # Alpha-Beta double excitation

                    k_alpha = removed_alpha[0]
                    k_beta = removed_beta[0] + norbs
                    l_alpha = added_alpha[0]
                    l_beta = added_beta[0]+ norbs
                    return  g_mo[k_alpha, k_beta, l_alpha, l_beta] # Correct for physicist's notation!

                else:
                    return 0
            else:
                return 0.0


        @staticmethod
        def slater_condon_two_body_kinda(total_diff, d_mo, lang_firsov_params_matrix, det1, det2, norbs):
            alpha1, beta1 = det1
            alpha2, beta2 = det2

            # Find the differences between alpha and beta orbitals separately
            diff_alpha = list(set(alpha1).symmetric_difference(set(alpha2)))
            diff_beta = list(set(beta1).symmetric_difference(set(beta2)))

            same_alpha = list(set(alpha1).intersection(set(alpha2)))
            same_beta = list(np.array(list(set(beta1).intersection(set(beta2))) )+ norbs)
            # print("same alpha: ", same_alpha,)
            # print("same beta: ", same_beta)

            same_in_both = set(same_alpha).union(set(same_beta))

            if total_diff == 0:

                # Diagonal element
                one_body  =  torch.tensor(0.0, dtype=torch.float64, requires_grad=False)

                for i in alpha1:
                    lambda_sum  =  torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                    for r in same_in_both:
                        if r != i:
                            lambda_sum = lambda_sum + lang_firsov_params_matrix[r,r]

                    one_body = one_body + d_mo[i,i] * lambda_sum
                    

                for j in beta1:
                    j_ = j+norbs
                    lambda_sum  =  torch.tensor(0.0, dtype=torch.float64, requires_grad=False)
                    for r in same_in_both:
                        if r != j_:
                            lambda_sum = lambda_sum + lang_firsov_params_matrix[r,r]
                    one_body = one_body + d_mo[j_,j_] * lambda_sum

                return one_body 


            elif total_diff == 2:

                one_body  =  torch.tensor(0.0, dtype=torch.float64, requires_grad=False)

                # Find the orbitals that differ
                if len(diff_alpha) == 2:
                    k, l = diff_alpha
                elif len(diff_beta) == 2:
                    k, l = diff_beta
                    k =k +norbs
                    l = l + norbs
                else:
                    k, l = diff_alpha[0], diff_beta[0]
                    l = l+norbs
                #print("k: ", k)
                #print("l: ", l)
                # Ensure k is in det1 and l is in det2
                orbs1 = alpha1 + tuple([b+norbs for b in beta1])
                orbs2 = alpha2 + tuple([b+norbs for b in beta2])
            
                beta1 = tuple([b+norbs for b in beta1])
                beta2 =tuple([b+norbs for b in beta2])
                # Ensure k is in det1 and l is in det2
                if k not in alpha1 + beta1 or l not in  alpha2 + beta2:
                    k, l = l, k


                for r in same_in_both:
                    if r != k and r!= l:
                        one_body = one_body+  d_mo[k,l] * lang_firsov_params_matrix[r,r]
                return one_body
            else: 
                return 0.0


        @staticmethod
        def lambda_two_body(lambda_vec, det, norbs):
            alpha, beta = det
            all_occ = list(alpha) + [x + norbs for x in beta]  # Combine alpha and beta orbitals
            two_body = torch.tensor(0.0, dtype=torch.float64, requires_grad=False)

            for i in all_occ:
                for j in all_occ:
                    if i != j:
                        two_body =  two_body + lambda_vec[i] * lambda_vec[j]

            return two_body



if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("hi")
