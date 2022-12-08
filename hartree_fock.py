import numpy as np
from numpy import linalg as la
from scipy import integrate
from scipy.special import erf
from matplotlib import pyplot as plt
plt.style.use('seaborn')


class hf_integrals:
    """
        A class to calculate the integrals needed to carry out Hartree-Fock.
        See Modern Quantum Chemistry : Introduction to Advanced Electronic Structure Theory
        by Szabo and Ostlund for meaning of integral notation (specifically, these
        formulas are found in appendix A).

        Attributes:
            basis_coeffs (list): STO-3G basis coefficients
    """
    def __init__(self, basis_coeffs):
        """
            Constructor. Accepts a list of basis coefficients and saves them to
            be used in the integral calculations below.
        """
        self.basis_coeffs = basis_coeffs

    def overlap_primitive(self, R_A, alpha_1, R_B, alpha_2):
        """
            Computes the integral (A|B) for two primitive gaussian functions.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                alpha_1 float: exponent coefficient of the first basis function
                R_B (..., 3) array: location of the second nucleus
                alpha_2 float: exponent coefficient of the second basis function

            Returns:
                The value of (A|B) for the given parameters.
        """
        factor_1 = (np.pi/(alpha_1 + alpha_2))**(3/2)
        factor_2 = np.exp(-alpha_1*alpha_2/(alpha_1 + alpha_2)*la.norm(R_A - R_B)**2)
        factor_3 = (2*alpha_1/np.pi)**(3/4)
        factor_4 = (2*alpha_2/np.pi)**(3/4)
        return factor_1*factor_2*factor_3*factor_4

    def overlap(self, R_A, R_B):
        """
            Computes the overlap matrix S using the overlap_primitive function.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                R_B (..., 3) array: location of the second nucleus

            Returns:
                S (..., 2, 2) array: the overlap matrix for these nucleus locations
        """
        S = []
        for center_1 in [R_A, R_B]:
            for center_2 in [R_A, R_B]:
                integral = 0
                for gaussian_1 in self.basis_coeffs:
                    for gaussian_2 in self.basis_coeffs:
                        integral += gaussian_1[0]*gaussian_2[0]* \
                            self.overlap_primitive(center_1, gaussian_1[1], center_2, gaussian_2[1])
                S.append(integral)
        S = np.array(S).reshape(2, 2)
        return S

    def F0(self, t):
        """
            Evaluates the 1-D gaussian integral defined in the text, using erf.
        """
        if t != 0:
            return 1/2*(np.pi/t)**(1/2)*erf(t**(1/2))
        else:
            return 1

    def kinetic_primitive(self, R_A, alpha_1, R_B, alpha_2):
        """
            Computes the integral (A|.5 grad^2|B) for two primitive gaussian functions.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                alpha_1 float: exponent coefficient of the first basis function
                R_B (..., 3) array: location of the second nucleus
                alpha_2 float: exponent coefficient of the second basis function

            Returns:
                The value of (A|.5 grad^2|B) for the given parameters.
        """
        norm_sq = la.norm(R_A - R_B)**2
        factor_1 = alpha_1*alpha_2/(alpha_1 + alpha_2)
        factor_2 = (3 - 2*factor_1*norm_sq)
        factor_3 = (np.pi/(alpha_1 + alpha_2))**(3/2)
        factor_4 = np.exp(-factor_1*norm_sq)
        factor_5 = (2*alpha_1/np.pi)**(3/4)
        factor_6 = (2*alpha_2/np.pi)**(3/4)
        return factor_1*factor_2*factor_3*factor_4*factor_5*factor_6

    def nuclear_attraction_primitive(self, R_A, alpha_1, R_B, alpha_2, R_C):
        """
            Computes the integral (A|Z_C / r_1C|B) for two primitive gaussian functions.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                alpha_1 float: exponent coefficient of the first basis function
                R_B (..., 3) array: location of the second nucleus
                alpha_2 float: exponent coefficient of the second basis function
                R_C float: location of the third nucleus (for hydrogen, will be either R_A or R_B)

            Returns:
                The value of (A|Z_C / r_1C|B) for the given parameters.
        """
        Z_C=1 #charge for hydrogen nucleus
        R_P = (alpha_1*R_A + alpha_2*R_B)/(alpha_1 + alpha_2)
        factor_1 = -2*np.pi*Z_C/(alpha_1 + alpha_2)
        factor_2 = np.exp(-alpha_1*alpha_2/(alpha_1 + alpha_2)*la.norm(R_A - R_B)**2)
        factor_3 = self.F0((alpha_1 + alpha_2)*la.norm(R_P - R_C)**2)
        factor_4 = (2*alpha_1/np.pi)**(3/4)
        factor_5 = (2*alpha_2/np.pi)**(3/4)
        return factor_1*factor_2*factor_3*factor_4*factor_5

    def core_hamiltonian(self, R_A, R_B):
        """
            Computes the core Hamiltonian matrix using the kinetic_primitive and
            nuclear_attraction_primitive functions.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                R_B (..., 3) array: location of the second nucleus

            Returns:
                H (..., 2, 2) array: the core Hamiltonian matrix for these nucleus locations
        """
        # kinetic energy calculations
        T = []
        for center_1 in [R_A, R_B]:
            for center_2 in [R_A, R_B]:
                integral = 0
                for gaussian_1 in self.basis_coeffs:
                    for gaussian_2 in self.basis_coeffs:
                        integral += gaussian_1[0]*gaussian_2[0]* \
                            self.kinetic_primitive(center_1, gaussian_1[1], center_2, gaussian_2[1])
                T.append(integral)
        T = np.array(T).reshape(2, 2)

        # nuclear attraction calculations
        V1 = []
        V2 = []
        for center_1 in [R_A, R_B]:
            for center_2 in [R_A, R_B]:
                integral_1 = 0
                integral_2 = 0
                for gaussian_1 in self.basis_coeffs:
                    for gaussian_2 in self.basis_coeffs:
                        integral_1 += gaussian_1[0]*gaussian_2[0]* \
                            self.nuclear_attraction_primitive(center_1, gaussian_1[1], center_2, gaussian_2[1], R_A)
                        integral_2 += gaussian_1[0]*gaussian_2[0]* \
                            self.nuclear_attraction_primitive(center_1, gaussian_1[1], center_2, gaussian_2[1], R_B)
                V1.append(integral_1)
                V2.append(integral_2)
        V1 = np.array(V1).reshape(2, 2)
        V2 = np.array(V2).reshape(2, 2)

        # now put together the core-Hamiltonian matrix
        H_core = T + V1 + V2
        return H_core

    def two_electron_primitive(self, R_A, alpha_1, R_B, alpha_2, R_C, alpha_3, R_D, alpha_4):
        """
            Computes the integral (AB|CD) for four primitive gaussian functions.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                alpha_1 float: exponent coefficient of the first basis function
                R_B (..., 3) array: location of the second nucleus
                alpha_2 float: exponent coefficient of the second basis function
                R_C (..., 3) array: location of the third nucleus
                alpha_3 float: exponent coefficient of the third basis function
                R_D (..., 3) array: location of the fourth nucleus
                alpha_4 float: exponent coefficient of the fourth basis function
            Returns:
                The value of (AB|CD) for the given parameters.
        """
        R_P = (alpha_1*R_A + alpha_2*R_B)/(alpha_1 + alpha_2)
        R_Q = (alpha_3*R_C + alpha_4*R_D)/(alpha_3 + alpha_4)
        norm_sq1 = la.norm(R_A - R_B)**2
        norm_sq2 = la.norm(R_C - R_D)**2
        norm_sq3 = la.norm(R_P - R_Q)**2
        factor_1 = 2*np.pi**(5/2)/((alpha_1 + alpha_2)*(alpha_3 + alpha_4)*np.sqrt(alpha_1 + alpha_2 + alpha_3 + alpha_4))
        factor_2 = np.exp(-alpha_1*alpha_2/(alpha_1 + alpha_2)*norm_sq1)
        factor_3 = np.exp(-alpha_3*alpha_4/(alpha_3 + alpha_4)*norm_sq2)
        factor_4 = self.F0((alpha_1 + alpha_2)*(alpha_3 + alpha_4)/(alpha_1 + alpha_2 + alpha_3 + alpha_4)*norm_sq3)
        factor_5 = (2*alpha_1/np.pi)**(3/4)
        factor_6 = (2*alpha_2/np.pi)**(3/4)
        factor_7 = (2*alpha_3/np.pi)**(3/4)
        factor_8 = (2*alpha_4/np.pi)**(3/4)
        return factor_1*factor_2*factor_3*factor_4*factor_5*factor_6*factor_7*factor_8

    def two_electron(self, R_A, R_B, R_C, R_D):
        """
            Evaluates the two electron integral for the given nuclei locations for the
            actual basis functions by summing up the results from different evaluations
            of two_electron_primitive.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                R_B (..., 3) array: location of the second nucleus
                R_C (..., 3) array: location of the third nucleus
                R_D (..., 3) array: location of the fourth nucleus

            Returns:
                The two electron integral for the given nuclei locations for the actual
                basis functions.
        """
        integral = 0
        for gaussian_1 in self.basis_coeffs:
            for gaussian_2 in self.basis_coeffs:
                for gaussian_3 in self.basis_coeffs:
                    for gaussian_4 in self.basis_coeffs:
                        integral += gaussian_1[0]*gaussian_2[0]*gaussian_3[0]*gaussian_4[0]* \
                            self.two_electron_primitive(R_A, gaussian_1[1], R_B, gaussian_2[1], \
                                R_C, gaussian_3[1], R_D, gaussian_4[1])
        return integral


class hf_solver:
    """
        Class for carrying out the Hartree-Fock (or self-consistent field) procedure.

        Attributes:
            basis_coeffs (iterable): STO-3G basis coefficients
            internuclear_dist float: distance between hydrogen nuclei (initialized to None)
            S (..., 2, 2) array: overlap matrix (initialized to None)
            H_core (..., 2, 2) array): core Hamiltonian matrix (initialized to None)
            two_electron_vals dict: dictionary of two electron integral values (initialized to None)
            P ..., 2, 2) array: density matrix (initialized to None)
            F ..., 2, 2) array: Fock matrix (initialized to None)
            C ..., 2, 2) array: FC = SC eps (initialized to None)
            eps ..., 2) array: FC = sc eps (initialized to None)
    """
    def __init__(self, basis_coeffs):
        """
            Constructor. Accepts a list of basis coefficients and saves them, and
            initializes all other attributes to None.
        """
        self.basis_coeffs = basis_coeffs
        self.internuclear_dist = None
        self.S = None
        self.H_core = None
        self.two_electron_vals = None
        self.P = None
        self.F = None
        self.C = None
        self.eps = None

    def compute_integrals(self, R_A, R_B):
        """
            Uses the hf_integrals class above to evaluate all the integrals we will
            need for Hartree-Fock for the Hydrogen molecule, and saves the as attributes.

            Parameters:
                R_A (..., 3) array: location of the first nucleus
                R_B (..., 3) array: location of the second nucleus
        """
        self.internuclear_dist = la.norm(R_A - R_B)
        integrator = hf_integrals(self.basis_coeffs)
        self.S = integrator.overlap(R_A, R_B)
        self.H_core = integrator.core_hamiltonian(R_A, R_B)
        two_electron_vals = dict()
        index_map = {'1':R_A, '2':R_B}
        for i1 in ['1', '2']:
            for i2 in ['1', '2']:
                for i3 in ['1', '2']:
                    for i4 in ['1', '2']:
                        two_electron_vals[i1 + i2 + i3 + i4] = \
                            integrator.two_electron(index_map[i1], index_map[i2], index_map[i3], index_map[i4])
        self.two_electron_vals = two_electron_vals

    def SCF(self, P0=None, tol=1e-10, maxiter=None):
        """
            Function to carry out Hartree-Fock in the two nucleus, two electron
            case for hydrogen. Note: this should probably have some guardrails
            (as in error catching), but just don't run this function until after
            running compute_integrals and it will be fine.

            Parameters:
                P0 (..., 2, 2) array: initial guess for density matrix
                tol (float): convegence criterion (optional)
                maxiter(int): maximum number of iterations (optional)
        """
        # diagonalize S and use it to get X (step 3 in szabo-ostlund pg 271)
        s, U = la.eig(self.S)
        X = U@np.diag(1/np.sqrt(s))@np.conjugate(U).T

        # guess an initial density matrix P and start iterating (step 4)
        if P0 is None: P0 = np.zeros((2, 2))
        counter = 0 #count iterations in case we want to cap them
        while True:
            # construct the Fock matrix F (steps 5 and 6)
            F = np.zeros((2, 2))
            for mu in range(1, 3):
                for nu in range(1, 3):
                    term = 0
                    term2 = 0
                    for lam in range(1, 3):
                        for sig in range(1, 3):
                            P_term = P0[lam - 1, sig - 1]
                            term += P_term*(self.two_electron_vals[str(mu) + str(nu) + str(sig) + str(lam)] \
                                - 1/2*self.two_electron_vals[str(mu) + str(lam) + str(sig) + str(nu)])
                    F[mu - 1, nu - 1] = self.H_core.copy()[mu - 1, nu - 1] + term

            # transform the Fock matrix and diagonalize (steps 7 and 8)
            F_p = X.T@F@X
            eps, C_p = la.eig(F_p)
            if eps[0] < eps[1]:
                eps = eps[::-1]
                C_p = np.concatenate((C_p[:, 1], C_p[:, 0])).reshape(2, 2)

            # calculate C using the C_p we got above (step 9)
            C = X@C_p

            # form a new density matrix P from C (step 10)
            P = np.ones((2, 2))
            for i in range(2):
                for j in range(2):
                    P[i, j] = 2*C[i, 1]*C[j, 1]

            # check convergence (step 11)
            std_dev = 0
            for i in range(2):
                for j in range(2):
                    std_dev += (P[i, j] - P0[i, j])**2
            if np.sqrt(std_dev) < 1e-4:
                break

            # optional iteration cap (defaults to no cap)
            if maxiter is not None and counter >= maxiter:
                raise Exception('Did not converge within maxiter iterations.')

            # set P0 = P for the next iteration, increment counter
            P0 = P
            counter += 1

        self.P = P
        self.F = F
        self.C = C
        self.eps = eps

    def compute_energy(self):
        """
            Computes the energy of the ground state of the hydrogen molecule
            for the internuclear distance used in the calculations above.
            Note: this should probably have some guardrails (as in error catching),
            but just don't run this function until after running both
            compute_integrals and SCF, and it will be fine.

            Returns:
                Total energy (ground state plus repulsion potential) of the molecule.
        """
        energy = 0
        for i in range(2):
            for j in range(2):
                energy += self.P[j, i]*(self.H_core[i, j] + self.F[i, j])
        energy /= 2
        repulsion = 1/self.internuclear_dist
        return energy + repulsion
