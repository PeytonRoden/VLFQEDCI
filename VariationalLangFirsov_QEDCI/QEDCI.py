
 

import itertools
import numpy as np
from .QEDHF import QED_HF


class CASCI:

  def __init__(self, mol_str, psi4_options_dict, options_dict):


    options_dict = {}


    #parse options dict
    self.omega = options_dict["omega"]
    self.coherent_state = options_dict["coherent_state"]
    self.photon_basis_size = options_dict["photon_basis_size"]
    self.lambda_vector = np.array(options_dict["lambda_vector"] )
    self.reference_type = options_dict["reference_type"] #qedhf or hf


    self.qedhf = QED_HF(mol_str, psi4_options_dict)
    qedhfdict = self.qedhf.qed_hf(self.lambda_vector)


    #I in physicists but we convert it to chemists
    self.ndocc = self.qedhf.ndocc

    #qedhf C
    self.C = self.qedhf.C
    self.C_ao= self.qedhf.C


    #regular C
    # self.C = self.qedhf.C_reg_HF
    # self.C_ao = self.qedhf.C_reg_HF


    self.I_ao = self.qedhf.I
    self.h_ao = self.qedhf.H_0
    self.d_ao = self.qedhf.d_ao
    self.q_ao = self.qedhf.q_ao
    self.d_n = self.qedhf.d_n
    self.d_two_body = self.qedhf.d_ao_two_body
    self.d_exp = self.qedhf.d_exp


    self.num_orbs = self.C.shape[0]
    print("norbs: " ,self.num_orbs)


    h_spinblock_ao = np.kron(np.eye(2), self.h_ao)
    d_spinblock_ao = np.kron(np.eye(2), self.d_ao)


    #scale off diag elements by 2
    q_spinblock_ao = np.kron(np.eye(2), self.q_ao)


    C_spinblock = np.kron(np.eye(2), self.C)
    I_spinblock_ao = CASCI.spin_block_tei(self.I_ao)
    #d_twobody_spinblock_ao = CASCI.spin_block_tei(d_ao_two_body)

    #antizymmetrized two electron integrals
    # <pr||qs> = <pr | qs> - <pr | sq>
    g_spinblock_ao = I_spinblock_ao.transpose(0, 2, 1, 3) - I_spinblock_ao.transpose(0, 2, 3, 1)
    # d_twobody_spinblock_ao_antisym  = d_twobody_spinblock_ao.transpose(0, 2, 1, 3) - d_twobody_spinblock_ao.transpose(0, 2, 3, 1)
    #I_spinblock_ao = CASCI.spin_block_tei(I).transpose(0, 2, 1, 3)

    Ca = self.C
    Cb = self.C
    self.C = np.block([
                [      Ca,         np.zeros_like(Cb)],
                [np.zeros_like(Ca),          Cb     ]])
    
    #self.I_mo = CASCI.ao_to_mo_tei(I_spinblock_ao, self.C)
    self.g_mo = CASCI.ao_to_mo_tei(g_spinblock_ao, self.C)

    self.h_mo = CASCI.ao_to_mo(h_spinblock_ao, self.C)
    self.q_mo = CASCI.ao_to_mo(q_spinblock_ao, self.C)

    #don't have to scale off diag elements by two for this one
    self.d_mo = CASCI.ao_to_mo(d_spinblock_ao, self.C)

    #d_spinblock_ao_tb = CASCI.spin_block_tei(d_ao_two_body) 
    d_spinblock_ao_tb = CASCI.spin_block_tei(self.d_two_body) 
    #print(d_spinblock_ao_tb)
    self.d_antisym_2_body =  d_spinblock_ao_tb.transpose(0, 2, 1, 3)  -  d_spinblock_ao_tb.transpose(0, 2, 3, 1)
    #print(self.d_antisym_2_body)
    self.d_antisym_2_body = CASCI.ao_to_mo_tei(self.d_antisym_2_body, self.C)



      
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
      V_transformed = np.einsum(
          'pqrs,pa,qb,rc,sd->abcd', gao, C.conj(), C.conj(), C, C, optimize=True)
      
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


  @staticmethod
  def slater_condon_one_body(total_diff, h_mo, det1, det2, norbs):
    alpha1, beta1 = det1
    alpha2, beta2 = det2


    # Find the differences between alpha and beta orbitals separately
    diff_alpha = list(set(alpha1).symmetric_difference(set(alpha2)))
    diff_beta = list(set(beta1).symmetric_difference(set(beta2)))

    if total_diff == 0:
        # print("identical")

        # Diagonal element
        one_body  = 0

        for i in alpha1:
            one_body += h_mo[i,i]
        for j in beta1:
            j_ = j+norbs
            one_body += h_mo[j_,j_]

     
        return one_body 



    elif total_diff == 2:

        # print("single excitation son ")

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
        if k not in alpha1 + beta1 or l not in  alpha2 + beta2:
            k, l = l, k

        one_body = 0
        one_body = h_mo[k,l]

        # Ensure k is in det1 and l is in det2
        if k not in alpha1 + beta1 or l not in  alpha2 + beta2:
            k, l = l, k
            
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

        two_body = 0

        for i in alpha1:
            for j in alpha2:
                two_body +=  0.5 * (g_mo[i, j, i, j])
        

        for i in np.array(beta1):
            for j in np.array(beta2):
                j_ = j+norbs
                i_ = i+norbs
                two_body += 0.5 * (g_mo[i_, j_, i_, j_] )

        for i in alpha1:
          for j in beta2:
              j_ = j+norbs
              two_body += 0.5 * (g_mo[i, j_, i, j_] )
        for i in alpha2:
          for j in beta1:              
              j_ = j+norbs
              two_body += 0.5 * (g_mo[i, j_, i, j_] )

     
        return two_body



    elif total_diff == 2:

        # print("single excitation son ")

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

        # Ensure k is in det1 and l is in det2
        if k not in alpha1 + beta1 or l not in  alpha2 + beta2:
            k, l = l, k

        two_body = 0
        

        for i in alpha1:
            if i != k:
                two_body += g_mo[k,i,l,i]

        for i in beta1:
            if i != k + norbs:
                i_ = i +norbs
                two_body += g_mo[k,i_,l,i_]


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
            return g_mo[k,m,l,n]

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

            return g_mo[k_alpha, k_beta, l_alpha, l_beta] # Correct for physicist's notation!

        else:
          return 0
    else:
        return 0.0



  def calculate_CI_energy(self, excitation_level, num_active_electrons, num_active_orbitals):


    options_dict= {}

    excitation_level = options_dict["excitation_level"]
    num_active_electrons = options_dict["num_active_electrons"]
    num_active_orbitals = options_dict["num_active_orbitals"]



    #first creeate reference and virtual orbitals for a system with 1 doubly occupied orbital and 3 virtual it looks likes this
    ### alpha occupied = [0] alpha virtual = [1,2,3]
    ### beta occupied = [0] beta virtual = [1,2,3]
    #determinant would look like this,  ([0,], [0,])

    norbs = self.C_ao.shape[0]
    ndocc = self.ndocc
    nvir = norbs - ndocc
    print("ndocc: ", ndocc)


    inactive_ref, active_ref, active_virtual = CASCI.generate_orbital_partitions(ndocc*2, num_active_electrons, num_active_orbitals)

    print("Inactive Reference:", inactive_ref)
    print("Active Reference:", active_ref)
    print("Active Virtual:", active_virtual)
    

    #build reference
    reference = []
    for i in range(ndocc):
        reference.append((i))

    reference = (reference,reference)
    #reference = tuple([tuple(reference[0]), tuple(reference[1])]) 

    virtual = []
    for i in range(ndocc, norbs):
        virtual.append(i)
    virtual = (virtual,virtual)

    basis = [tuple([tuple(reference[0]), tuple(reference[1])])   ]


    #we have to make sure singles don't couple to reference
    #singles = CASCI.generate_excited_determinants(reference, virtual, 1)
    singles = CASCI.generate_active_space_excited_determinants(active_reference=active_ref, active_virtual_orbitals=active_virtual, inactive_reference=inactive_ref, N= 1)

    excited_determinants = []
    for i in range( 1, excitation_level+1, 1):
      excited_determinants_new = CASCI.generate_active_space_excited_determinants(active_reference=active_ref, active_virtual_orbitals=active_virtual, inactive_reference=inactive_ref, N = i)
      excited_determinants =  excited_determinants + excited_determinants_new

    basis = basis + excited_determinants 
    basis= tuple(basis)

    el_basis_size = len(basis)

    #append photon state onto basis

    coupled_basis = []

    for ph_num in range(self.photon_basis_size):

        for i in range(len(basis)):
            coupled_basis.append(((basis[i]), (ph_num)))


    coupled_basis = tuple(coupled_basis)
    #print(coupled_basis)

    basis_size = len(coupled_basis)

    print("basis _size : ", basis_size)

    H_CI_e = np.zeros((basis_size, basis_size))

    H_CI_dse = np.zeros((basis_size, basis_size))

    H_CI_blc = np.zeros((basis_size, basis_size))

    H_CI_ph = np.zeros((basis_size, basis_size))

    a_dag, a = CASCI.create_creation_operator(self.photon_basis_size), CASCI.create_annihilation_operator(self.photon_basis_size)


    identity_photon = np.eye(self.photon_basis_size)
    identity_electronic = np.eye(el_basis_size)
    a_dag_mult_a = a_dag@a
    a_dag_plus_a = a_dag + a


    if self.coherent_state == True:
        #offset = self.slater_condon_one_body(self.h_mo, basis[-1], basis[-1]) + self.slater_condon_two_body(self.I_mo, basis[-1], basis[-1])
        for i in range(basis_size):
            for j in range(i, basis_size):

                #pick out phootonic and electronic parts of basis
                photon_basis_i = coupled_basis[i][1]
                photon_basis_j = coupled_basis[j][1]
                el_basis_i = coupled_basis[i][0]
                el_basis_j = coupled_basis[j][0]


                total_diff = CASCI.compute_diff(tuple(el_basis_i),tuple(el_basis_j))
                photon_diff = photon_basis_i - photon_basis_j
                #print(photon_diff)

                
                phase = 1

                if photon_diff == 0 or photon_diff==1 or photon_diff==-1:
                    if total_diff == 2 or total_diff == 4:
                        phase = CASCI.compute_phase_factor(tuple(el_basis_i),tuple(el_basis_j))
                    

                if False:
                    pass
                #everything else
                else:


                    val = 0
                    if photon_diff == 0:
                        #regular electronic hamiltonian part
                        if total_diff ==0 or total_diff==2 or total_diff ==4:
                            val = phase * (CASCI.slater_condon_one_body(total_diff, self.h_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) +  CASCI.slater_condon_two_body(total_diff, self.g_mo, tuple(el_basis_i),tuple(el_basis_j), norbs))
                        else:
                            val = 0

                    H_CI_e[i,j] = val
                    H_CI_e[j,i] = val

                    val = 0
                    #dse part

                    if photon_diff == 0:
                        if total_diff ==0 or total_diff==2 or total_diff ==4:
                            val = ( CASCI.slater_condon_two_body(total_diff, self.d_antisym_2_body, tuple(el_basis_i),tuple(el_basis_j), norbs) - 1 * CASCI.slater_condon_one_body(total_diff, self.q_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) )
                            val -= 2 * self.d_exp * CASCI.slater_condon_one_body(total_diff, self.d_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) 
                            if total_diff ==0:
                                val += self.d_exp **2
                            val *= phase
                        else:
                            val = 0


                    H_CI_dse[i,j] = val
                    H_CI_dse[j,i] = val


                    val = 0
                    #blc part
        
                    if photon_diff ==1 or photon_diff ==-1 or photon_diff == 0 :
                        if total_diff ==0 or total_diff==2:
                            val = ( CASCI.slater_condon_one_body(total_diff, self.d_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) )
                            if total_diff ==0:
                                val -= self.d_exp
                            
                            val *= phase 
                        else:
                            val = 0
                        val *= (a_dag_plus_a )[photon_basis_i, photon_basis_j]
                    
                    H_CI_blc[i,j] = val
                    H_CI_blc[j,i] = val

                    val = 0
                    #photon_part part
                    if total_diff ==0 and (photon_diff == 0 or photon_diff==1 or photon_diff==-1):
                        val = a_dag_mult_a[photon_basis_i, photon_basis_j]
                    else:
                        val = 0

                    H_CI_ph[i,j] = val
                    H_CI_ph[j,i] = val


    elif self.coherent_state == False:
        for i in range(basis_size):
            for j in range(i, basis_size):

                #pick out phootonic and electronic parts of basis
                photon_basis_i = coupled_basis[i][1]
                photon_basis_j = coupled_basis[j][1]
                el_basis_i = coupled_basis[i][0]
                el_basis_j = coupled_basis[j][0]


                total_diff = CASCI.compute_diff(tuple(el_basis_i),tuple(el_basis_j))
                photon_diff = photon_basis_i - photon_basis_j
                #print(photon_diff)

                
                phase = 1

                if photon_diff == 0 or photon_diff==1 or photon_diff==-1:
                    if total_diff == 2 or total_diff == 4:
                        phase = CASCI.compute_phase_factor(tuple(el_basis_i),tuple(el_basis_j))
                    
                if False:
                    pass
                else:
                    val = 0
                    if photon_diff == 0:
                        #regular electronic hamiltonian part
                        if total_diff ==0 or total_diff==2 or total_diff ==4:
                            val = phase * (CASCI.slater_condon_one_body(total_diff, self.h_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) +  CASCI.slater_condon_two_body(total_diff, self.g_mo, tuple(el_basis_i),tuple(el_basis_j), norbs))
                        else:
                            val = 0

                    H_CI_e[i,j] = val
                    H_CI_e[j,i] = val

                    val = 0
                    #dse part

                    if photon_diff == 0:
                        if total_diff ==0 or total_diff==2 or total_diff ==4:
                            val = ( CASCI.slater_condon_two_body(total_diff, self.d_antisym_2_body, tuple(el_basis_i),tuple(el_basis_j), norbs) - 1 * CASCI.slater_condon_one_body(total_diff, self.q_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) )
                            val += 2 * self.d_n * CASCI.slater_condon_one_body(total_diff, self.d_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) 
                            if total_diff ==0:
                                val += self.d_n **2
                            val *= phase
                        else:
                            val = 0

                    H_CI_dse[i,j] = val
                    H_CI_dse[j,i] = val

                    val = 0
                    #blc part
        
                    if photon_diff ==1 or photon_diff ==-1 or photon_diff == 0 :
                        if total_diff ==0 or total_diff==2:
                            val = ( CASCI.slater_condon_one_body(total_diff, self.d_mo, tuple(el_basis_i),tuple(el_basis_j), norbs) )
                            if total_diff ==0:
                                val += self.d_n
                            
                            val *= phase 
                        else:
                            val = 0
                        val *= (a_dag_plus_a )[photon_basis_i, photon_basis_j]
                    
                    H_CI_blc[i,j] = val
                    H_CI_blc[j,i] = val

                    val = 0
                    #photon_part part
                    if total_diff ==0 and (photon_diff == 0 or photon_diff==1 or photon_diff==-1):
                        val = a_dag_mult_a[photon_basis_i, photon_basis_j]
                    else:
                        val = 0

                    H_CI_ph[i,j] = val
                    H_CI_ph[j,i] = val



    H_PF =  H_CI_e + self.omega *  H_CI_ph  - np.sqrt(self.omega/2)*  H_CI_blc     +  0.5 *  H_CI_dse

    np.set_printoptions(precision = 7)

    vals ,vecs = np.linalg.eigh(H_PF)

    return vals,vecs, H_PF




if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("hi")
