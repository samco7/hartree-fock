import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn')

def plot_energy(distances, energy_vals, save_loc=None):
    z = np.zeros(10)
    x = np.linspace(-1, 6, 10)
    y = np.linspace(-.2, .5, 10)
    plt.figure()
    plt.plot(distances, np.array(energy_vals) + 2*(.4666))
    plt.plot(x, z, 'k', zorder=0)
    plt.plot(z, y, 'k')
    plt.xlabel(r'$R$ (a.u.)', fontsize=18)
    # plt.ylabel('Energy', fontsize=18)
    plt.legend([r'$E(H_2) - 2E(H)$'], fontsize=18)
    plt.grid(False)
    plt.xlim(-.01, 6)
    plt.ylim(-.2, .5)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()

def compare_slater_gaussian(save_loc=None):
    slater = lambda zeta, r: (zeta**3/np.pi)**(1/2)*np.exp(-zeta*np.abs(r))
    gaussian = lambda alpha, r: (2*alpha/np.pi)**(3/4)*np.exp(-alpha*np.abs(r)**2)

    r = np.linspace(0, 4, 100)

    plt.figure(figsize=(8, 4.8))
    plt.plot(r, slater(1, r))
    plt.plot(r, gaussian(.270950, r))
    plt.xlim(0, 4)
    plt.ylim(0, .56)
    plt.grid(False)
    plt.legend(['Slater', 'Gaussian'], fontsize=18)
    plt.xlabel('Radius (a.u.)', fontsize=18)
    plt.ylabel(r'$\phi_{1s}$', fontsize=18)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()

def compare_sto_ng(save_loc=None):
    slater = lambda zeta, r: (zeta**3/np.pi)**(1/2)*np.exp(-zeta*np.abs(r))
    gaussian = lambda alpha, r: (2*alpha/np.pi)**(3/4)*np.exp(-alpha*np.abs(r)**2)

    one_coeff = [(1, .270950*1.24**2)]
    two_coeffs = [(.4301284983, 1.309756377), (.6789135305, .2331359749)]
    three_coeffs = [(.4446345422, .1688554040), (.5353281423, .6239137298), (.1543289673, 3.425250914)]

    plt.figure(figsize=(8, 4.8))
    r = np.linspace(0, 4, 100)
    plt.plot(r, slater(1.24, r), c='k')
    for coeffs in [one_coeff, two_coeffs, three_coeffs]:
        cgf = 0
        for coeff in coeffs:
            cgf += coeff[0]*gaussian(coeff[1], r)
        plt.plot(r, cgf, '--')
    plt.xlabel('Radius (a.u.)', fontsize=18)
    plt.ylabel(r'$\phi_{1s}$', fontsize=18)
    plt.legend(['Slater', 'STO-1G', 'STO-2G', 'STO-3G'], fontsize=18)
    plt.grid(False)
    plt.xlim(0, 4)
    plt.ylim(0, .8)
    plt.tight_layout()
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()
