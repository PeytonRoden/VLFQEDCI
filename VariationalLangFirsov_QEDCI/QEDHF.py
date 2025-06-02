
import psi4
import numpy as np
import time

class QED_HF:

    def __init__(self, mol_str, psi_4_options_dict):
        psi4.set_memory(int(5e8))
        self.numpy_memory = 2

        np.set_printoptions(precision = 7)

        # Set output file
        psi4.core.set_output_file('output.dat', False)

        self.mol = psi4.geometry(mol_str)

        # Set computation options
        psi4.set_options(psi_4_options_dict)

        # run an RHF calculation with psi4 and save the wavefunction object and RHF energy
        self.scf_e, self.wfn = psi4.energy('scf', return_wfn=True)

    def qed_hf(self , lambda_vector):

        #lambda_vector = np.array([0.0,0.0,0.0])
        # create instance of the mintshelper class
        mints = psi4.core.MintsHelper(self.wfn.basisset())

        # Overlap matrix, S, and orthogonalization matrix, A
        S = mints.ao_overlap()
        A = mints.ao_overlap()
        A.power(-0.5, 1.0e-16)
        A = np.asarray(A)

        # Number of basis Functions & doubly occupied orbitals
        nbf = S.shape[0]
        ndocc = self.wfn.nalpha()

        print('Number of occupied orbitals: %3d' % (ndocc))
        print('Number of basis functions: %3d' % (nbf))

        # Memory check for ERI tensor
        I_size = (nbf**4) * 8.e-9
        print('\nSize of the ERI tensor will be {:4.2f} GB.'.format(I_size))
        if I_size > self.numpy_memory:
            psi4.core.clean()
            raise Exception("Estimated memory utilization (%4.2f GB) exceeds allotted memory \
                            limit of %4.2f GB." % (I_size, self.numpy_memory))

        # Build ERI Tensor
        I = np.asarray(mints.ao_eri())

        # Build core Hamiltonian for canonical HF theory
        # Kinetic energy matrix
        T = np.asarray(mints.ao_kinetic())
        # e-N attraction matrix
        V = np.asarray(mints.ao_potential())
        # canonical Core Hamiltonian
        H_0 = T + V

        # Prepare a guess density matrix from converged HF orbitals
        C = np.asarray(self.wfn.Ca())

        self.C_reg_HF = np.copy(C)

        print("C regular: ", self.C_reg_HF)
        
        # use canonical HF orbitals for guess of the HF-PF orbitals
        Cocc = C[:, :ndocc]
        # form guess density
        D = np.einsum("pi,qi->pq", Cocc, Cocc)

        # Prepare mu terms
        # nuclear dipole
        mu_nuc_x = self.mol.nuclear_dipole()[0]
        mu_nuc_y = self.mol.nuclear_dipole()[1]
        mu_nuc_z = self.mol.nuclear_dipole()[2]

        # electronic dipole integrals in AO basis
        mu_ao_x = np.asarray(mints.ao_dipole()[0])
        mu_ao_y = np.asarray(mints.ao_dipole()[1])
        mu_ao_z = np.asarray(mints.ao_dipole()[2])

        # d_ao = \lambda \cdot \mu_ao
        d_ao = lambda_vector[0] * mu_ao_x
        #print(d_ao)
        d_ao += lambda_vector[1] * mu_ao_y
        #print(d_ao)
        d_ao += lambda_vector[2] * mu_ao_z
        #print(d_ao)

        # compute electronic dipole expectation values
        mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
        mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
        mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)


        # Store the total RHF dipole moment which contains electronic and nuclear contribution
        rhf_dipole_moment = np.array([mu_exp_x + mu_nuc_x, mu_exp_y + mu_nuc_y, mu_exp_z + mu_nuc_z])

        # store <d> = \lambda \cdot <\mu>
        d_exp = lambda_vector[0] * mu_exp_x + lambda_vector[1] * mu_exp_y + lambda_vector[2] * mu_exp_z

        print(rhf_dipole_moment)

        # Prepare the Quadrupole terms
        Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
        Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
        Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
        Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
        Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
        Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

        # 1 electron quadrupole term
        q_ao = -0.5 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
        q_ao -= 0.5 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy 
        q_ao -= 0.5 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

        q_ao -= lambda_vector[0] * lambda_vector[1] * Q_ao_xy
        q_ao -= lambda_vector[0] * lambda_vector[2] * Q_ao_xz
        q_ao -= lambda_vector[1] * lambda_vector[2] * Q_ao_yz

        # O^{DSE}
        O_DSE = q_ao - d_exp * d_ao 


        # Constant quadratic dipole energy term
        d_c = 0.5 * d_exp ** 2

        # Core Hamiltonian including 1-electron DSE term
        H = H_0 + O_DSE

        mu_n = np.array([mu_nuc_x, mu_nuc_y, mu_nuc_z])
        d_n = np.dot(mu_n, lambda_vector) 
        d_n_sq = np.dot(mu_n, lambda_vector) * np.dot(mu_n, lambda_vector) 

        print("\nStart SCF iterations:\n")
        t = time.time()
        E = 0.0
        Enuc = self.mol.nuclear_repulsion_energy()
        Eold = 0.0
        E_1el_crhf = np.einsum("pq,pq->", H_0 + H_0, D)
        E_1el = np.einsum("pq,pq->", H + H, D)
        print("Canonical RHF One-electron energy = %4.16f" % E_1el_crhf)
        print("CQED-RHF One-electron energy      = %4.16f" % E_1el)
        print("Nuclear repulsion energy          = %4.16f" % Enuc)
        print("Dipole energy                     = %4.16f" % d_c)

        # Set convergence criteria for energy and density
        E_conv = 1.0e-10
        D_conv = 1.0e-10




        t = time.time()
        # maxiter
        maxiter = 500
        for SCF_ITER in range(1, maxiter + 1):

            # Canonical 2ERI contribution to Fock matrix - Eq. (15)
            J = np.einsum("pqrs,rs->pq", I, D)
            K = np.einsum("prqs,rs->pq", I, D)

            # Pauli-Fierz 2-e dipole-dipole terms - Eq. (15)
            M = np.einsum("pq,rs,rs->pq", d_ao, d_ao, D)
            N = np.einsum("pr,qs,rs->pq", d_ao, d_ao, D)

            # PF-Fock Matrix
            F = H + J * 2 - K  + 2 * M - N



            ### Check Convergence of the Density Matrix
            diis_e = np.einsum("ij,jk,kl->il", F, D, S) - np.einsum("ij,jk,kl->il", S, D, F)
            diis_e = A.dot(diis_e).dot(A)
            dRMS = np.mean(diis_e ** 2) ** 0.5

            # SCF energy and update
            SCF_E = np.einsum("pq,pq->", F + H, D) + Enuc + d_c

            print(
                "SCF Iteration %3d: Energy = %4.16f   dE = % 1.5E   dRMS = %1.5E"
                % (SCF_ITER, SCF_E, (SCF_E - Eold), dRMS)
            )
            if (abs(SCF_E - Eold) < E_conv) and (dRMS < D_conv):
                break

            Eold = SCF_E

            # Diagonalize Fock matrix
            Fp = A.dot(F).dot(A)  
            e, C2 = np.linalg.eigh(Fp) 
            # Back transform
            C = A.dot(C2)  
            Cocc = C[:, :ndocc]
            # update density
            D = np.einsum("pi,qi->pq", Cocc, Cocc)  

            # update electronic dipole expectation value
            mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
            mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
            mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

            # update \lambda \cdot <\mu>
            d_exp = (
                lambda_vector[0] * mu_exp_x
                + lambda_vector[1] * mu_exp_y
                + lambda_vector[2] * mu_exp_z
            )
            
            # update 1-electron DSE term
            O_DSE = q_ao  -   d_exp * d_ao 


            # update Core Hamiltonian
            H = H_0   + O_DSE

            # update constant dipole energy
            d_c = 0.5 * d_exp ** 2

            if SCF_ITER == maxiter:
                psi4.core.clean()
                raise Exception("Maximum number of SCF cycles exceeded.")
        print("Total time for SCF iterations: %.3f seconds \n" % (time.time() - t))
        print("QED-HF   energy: %.8f hartree" % SCF_E)
        print("Psi4  SCF energy: %.8f hartree" % self.scf_e)

        #save qed-hf energy: 
        self.QEDHF_E = SCF_E



        # electronic dipole integrals in AO basis
        mu_ao_x = np.asarray(mints.ao_dipole()[0])
        mu_ao_y = np.asarray(mints.ao_dipole()[1])
        mu_ao_z = np.asarray(mints.ao_dipole()[2])

        # d_ao = \lambda \cdot \mu_ao
        d_ao_x = lambda_vector[0] * mu_ao_x
        d_ao_y = lambda_vector[1] * mu_ao_y
        d_ao_z = lambda_vector[2] * mu_ao_z

        d_ao =  d_ao_x+d_ao_y+d_ao_z

        d_ao_two_body = 2 * np.einsum('pq,rs -> pqrs', d_ao , d_ao)

        # Prepare the Quadrupole terms
        Q_ao_xx = np.asarray(mints.ao_quadrupole()[0])
        Q_ao_xy = np.asarray(mints.ao_quadrupole()[1])
        Q_ao_xz = np.asarray(mints.ao_quadrupole()[2])
        Q_ao_yy = np.asarray(mints.ao_quadrupole()[3])
        Q_ao_yz = np.asarray(mints.ao_quadrupole()[4])
        Q_ao_zz = np.asarray(mints.ao_quadrupole()[5])

        q_ao = -1 * lambda_vector[0] * lambda_vector[0] * Q_ao_xx
        q_ao -= 1 * lambda_vector[1] * lambda_vector[1] * Q_ao_yy 
        q_ao -= 1 * lambda_vector[2] * lambda_vector[2] * Q_ao_zz

        q_ao -= 2* lambda_vector[0] * lambda_vector[1] * Q_ao_xy
        q_ao -= 2* lambda_vector[0] * lambda_vector[2] * Q_ao_xz
        q_ao -= 2* lambda_vector[1] * lambda_vector[2] * Q_ao_yz


        #recalculate d_exp
        Cocc = C[:, :ndocc]
        # update density
        D = np.einsum("pi,qi->pq", Cocc, Cocc)  

        # update electronic dipole expectation value
        mu_exp_x = np.einsum("pq,pq->", 2 * mu_ao_x, D)
        mu_exp_y = np.einsum("pq,pq->", 2 * mu_ao_y, D)
        mu_exp_z = np.einsum("pq,pq->", 2 * mu_ao_z, D)

        # update \lambda \cdot <\mu>
        d_exp = (
            lambda_vector[0] * mu_exp_x
            + lambda_vector[1] * mu_exp_y
            + lambda_vector[2] * mu_exp_z
        )

        dipole_matrix = d_ao
        #convert it to mo basis
        dipole_matrix =  C.T @ d_ao @ C
        d_vals, d_vecs = np.linalg.eigh(dipole_matrix)
        self.d_vecs = d_vecs
        self.d_eigvals = d_vals
        self.C_dipole = C @ d_vecs

        self.d_exp = -d_exp #d expectation value
        self.C = C #coefficient matrix from qedhf
        self.d_n = -d_n #nuclear dipole
        self.d_ao = d_ao *  -1 #one body dipole matrix terms in ao basis
        self.d_ao_two_body = d_ao_two_body * 1 #two body diole tensor terms in ao basis
        self.q_ao = q_ao * -1 #quadrople matrix terms in ao basis
        self.ndocc = ndocc #number of doubly occupied molecular robitals
        self.I = I #ERI tensor in ao basis
        self.H_0 = H_0 #kinetic plus potential integrals in ao basis


        qedhfdict = {
                "d_exp": self.d_exp,
                "C": self.C,
                "C_reghf": self.C_reg_HF,
                "C_qedhf_dipole_basis": self.C_dipole,
                "d_n": self.d_n,
                "d_ao": self.d_ao,
                "d_ao_two_body": self.d_ao_two_body,
                "q_ao": self.q_ao,
                "ndocc": self.ndocc,
                "I": self.I,
                "H_0": self.H_0,

                }


        
        return qedhfdict

 


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print("hi")
