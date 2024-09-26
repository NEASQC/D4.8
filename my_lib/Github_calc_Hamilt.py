#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pickle
from pyscf import gto, scf, ao2mo, ci
from functools import reduce
from qat.fermion import ElectronicStructureHamiltonian
from qat.fermion.chemistry.ucc import transform_integrals_to_new_basis
from qat.fermion.chemistry.ucc import select_active_orbitals, convert_to_h_integrals, compute_active_space_integrals
import scipy


def ob_tb_integ(mol : gto.mole.Mole, m_mol: scf.hf.RHF):
    """ Computes one-body integral and two-body integral.

    Input : 
        - mol : PySCF molecule
        - m_mol : PySCF mean-field molecule

    Output : 
        - one_body_integ : one-body integral
        - two_body_integ : two-body integral

    """
    
    ########
    # 1 : calculation of one_body_integral
    ########
    n_orbitals = m_mol.mo_coeff.shape[1]
    one_body_compressed = reduce(np.dot, (m_mol.mo_coeff.T, m_mol.get_hcore(), m_mol.mo_coeff))
    one_body_integ = one_body_compressed.reshape(n_orbitals, n_orbitals).astype(float)

    ########
    # 2 : calculation of two_body_integral
    ########
    two_body_compressed = ao2mo.kernel(mol, m_mol.mo_coeff)
    two_body_integ = ao2mo.restore(1, two_body_compressed, n_orbitals)
    two_body_integ = np.asarray(two_body_integ.transpose(0, 2, 3, 1), order='C')

    return one_body_integ, two_body_integ


def get_active_space_hamiltonian(one_body_integrals: np.array, two_body_integrals: np.array, 
                                 noons: list, nels: int, 
                                 nuclear_repulsion: float, 
                                 threshold_1: float, threshold_2: float):
    """ Creates active space Hamiltonian, with respect to the following rule :
            if occupation number <= 2 - threshold_1 :
                -> orbital is frozen
            elif occupation number < threshold_2 :
                -> orbital is virtual
            else:
                -> orbital is active
    
    Input :
        - one_body_integ : one-body integral
        - two_body_integ : two-body integral
        - noons : list of natural orbital occupation numbers
        - nels : total number of electrons
        - nuclear_repulsion : nuclear repulsion energy
        - threshold_1 : threshold to determine frozen orbitals
        - threshold_2 : threshold to determine virtual orbitals
    
    Output :
        - H_active : active space Hamiltonian
        - active_inds : list of indices of active orbitals
        - occ_inds : list of indices of occupied orbitals
    """

    active_inds, occ_inds = select_active_orbitals(noons, nels, 
                                                   threshold_1, threshold_2)
    
    c_act, Ipq_act, Ipqrs_act = compute_active_space_integrals(one_body_integrals, two_body_integrals, 
                                                               active_inds, occ_inds)
    
    hpq, hpqrs = convert_to_h_integrals(Ipq_act, Ipqrs_act)

    H_active = ElectronicStructureHamiltonian(hpq=hpq, hpqrs=hpqrs, 
                                              constant_coeff = nuclear_repulsion+c_act)  
    
    return H_active, active_inds, occ_inds


def H_with_active_space_reduction(one_body_integ: np.array, two_body_integ: np.array, 
                                  mol: gto.mole.Mole, m_mol: scf.hf.RHF,
                                  nb_homo: int, nb_lumo: int):
    """ Creates active space Hamiltonian from number of HOMO/LUMO orbitals required.

    Input : 
        - one_body_integ : one-body integral
        - two_body_integ : two-body integral
        - mol : PySCF molecule
        - m_mol : PySCF mean-field molecule
        - nb_homo : number of frozen orbitals
        - nb_lumo : number of virtual orbitals
    
    Output :
        - H_active : Hamiltonian after active space reduction (orbital freezing)
        - active_inds : list of indices of active orbitals
        - occ_inds :  list of indices of occupied orbitals
        - noons : list of natural-orbital occupation numbers 
        - orbital energies : list of energies of each molecular orbital
        - nels : total number of electrons
        - eps1 : threshold to determine frozen orbitals
        - eps2 : threshold to determine virtual orbitals
    """
    
    ########
    # 1 : preparation
    ########
    nels = mol.nelectron
    nuclear_repulsion = mol.energy_nuc()
    orbital_energies = m_mol.mo_energy
    
    ci_mol = ci.CISD(m_mol.run()).run()
    rdm1 = ci_mol.make_rdm1()
    noons, basis_change = np.linalg.eigh(rdm1)
    noons = list(reversed(noons))
    basis_change = np.flip(basis_change, axis=1)
    
    ## This has been removed for accelerating calculation (procedure O(N^8) too expensive for large systems...)
#     one_body_integrals, two_body_integrals = transform_integrals_to_new_basis(one_body_integ,
#                                                                               two_body_integ,
#                                                                               basis_change)
    one_body_integrals = one_body_integ
    two_body_integrals = two_body_integ
    
    
    ########
    # 2 : calculation of the reduced Hamiltonian
    ########
    assert nb_homo >= 0, 'nb_homo >= 0'
    assert nb_lumo >= 0, 'nb_homo >= 0'
    assert nb_homo <= nels//2, f'nb_homo <= {nels//2}'
    assert nb_lumo <= len(noons)-nels//2, f'nb_lumo <= {len(noons)-nels//2}'

    homo_min = nels//2-nb_homo
    lumo_max = nels//2 + nb_lumo
    if homo_min == 0:
        eps1 = 0
    else:
        eps1 = 2 - (noons[homo_min-1]+noons[homo_min])/2
    eps2 = noons[lumo_max-1]
    
    
    H_active, active_inds, occ_inds = get_active_space_hamiltonian(one_body_integrals, two_body_integrals, 
                                                                   noons, nels, 
                                                                   nuclear_repulsion, 
                                                                   threshold_1 = eps1, threshold_2 = eps2)
    
    return H_active, active_inds, occ_inds, noons, orbital_energies, nels, eps1, eps2

########################
### Hamiltonian save ###
########################

def save_H_into_dict(l1: float, save_filename: str, 
                     mol: gto.mole.Mole, m_mol: scf.hf.RHF, 
                     nb_homo: int, nb_lumo: int, 
                     calc_E_exact: bool = False):
    """ Full method for calculating an active space Hamiltonian and saving all data in a dictionnary.

    Input :
        - l1 : varying parameter (e.g. : bond length of a molecule)
        - save_filename : data will be saved in 'save_filename.H.pickle'
        - mol : PySCF molecule
        - m_mol : PySCF mean-field molecule
        - nb_homo : number of frozen orbitals
        - nb_lumo : number of virtual orbitals
        - calc_E_exact : True if exact energy must be calculated
        
    Output : 
        - dic_H_save : dictionnary contained in save_filename.H.pickle, with
            * 1st key : l1 (varying parameter)
            * 2nd key : chemical basis set of the molecule
            * 3rd key : nb_homo (characterizes the active space reduction)
            * 4th key : nb_lumo (characterizes the active space reduction)
            * Then :
                - H_active : Hamiltonian after active space reduction (orbital freezing)
                - active_inds, occ_inds :  list of index of active/occupied orbitals
                - noons : list of natural-orbital occupation numbers 
                - orbital energies : list of energies of each molecular orbital
                - nels : total number of electrons
                - E_exact : exact ground state energy of the system, obtained with diagonalization (None if calc_E_exact == False)
    """

    ########
    # 1 : search if the file already exists
    ########
    try:
        with open(f'{save_filename}.H.pickle','rb') as f1:
            dic_H_save = pickle.load(f1)
    except:
        print(f'Error : The dictionary {save_filename}.pickle doesn\'t exist.\n => Creation of a new one')
        dic_H_save = {}
    try:
        dic_H_save[str(l1)]
    except:
        dic_H_save[str(l1)] = {}
        
    
    ########
    # 2 : calculation of one-body and two-body integrals
    ########
    ob, tb = ob_tb_integ(mol,m_mol)
    print(f'Nb of qubits (before reduction) : {2*ob.shape[0]}')
    
    ########
    # 3 : active space selection
    ########
    H_active, active_inds, occ_inds, noons, orbital_energies, nels, eps1, eps2 = H_with_active_space_reduction(ob, tb, 
                                                                                                               mol, m_mol, 
                                                                                                               nb_homo, nb_lumo)

    print(f'··· nb_homo = {nb_homo} | nb_lumo = {nb_lumo}')
    print(f'··· noons = {noons}')
    print(f'··· occ_inds = {occ_inds}')
    print(f'··· active_inds = {active_inds}')
    print(f'Nb of qubits (after reduction) : {H_active.nbqbits}')
    
    ########
    # 4 : creation and/or update of dictionnary for saving
    ########
    try:
        dic_H_save[str(l1)][mol.basis]
    except:
        dic_H_save[str(l1)][mol.basis] = {}
    try:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)]
    except:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)] = {}
    try:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]
    except:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)] = {}
        
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['H_active'] = H_active
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['active_inds'] = active_inds
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['occ_inds'] = occ_inds
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['noons'] = noons
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['orbital_energies'] = orbital_energies
    dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['nels'] = nels

    ########
    # 5 : Exact energy calculation if required
    ########
    if calc_E_exact == True:
        try:
            EIGVAL, _ = scipy.sparse.linalg.eigs(H_active.get_matrix(sparse=True))
            E_exact = np.min(np.real(EIGVAL))
            dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['E_exact'] = E_exact
        except Exception as e_print_sparse:
            print(f'Exception (sparse diag) : {e_print_sparse}')
            dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['E_exact'] = None
    else:
        dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['E_exact'] = None
    
    ########
    # 6 : Save the dictionnary into save_filename.H.pickle
    ########
    with open(f'{save_filename}.H.pickle','wb') as f2:
        pickle.dump(dic_H_save,f2)
        print(f'=> The dictionary of results is saved in {save_filename}.H.pickle')

    return dic_H_save    
    
##############################
### Distortions of benzene ###
##############################

def build_benz_dist_1(alpha, basis='sto-3g'):
    """
    Input : alpha (varying parameter), basis set
    Ouput : benzene PySCF molecule under distortion 1 (according to paper ...)
    """
    mol_benz = gto.Mole()
    mol_benz.verbose = 5
    mol_benz.output = 'benz.log'
    R_CC = 1.39*alpha
    R_CH = 1.09
    c_h = ''
    for i in range(6):
        angle = np.pi/6 + i*np.pi/3
        x, y = R_CC*np.cos(angle), R_CC*np.sin(angle)
        x_H, y_H = (R_CC+R_CH)*np.cos(angle), (R_CC+R_CH)*np.sin(angle)
        c_h += f'C {x} {y} 0; H {x_H} {y_H} 0;'
    mol_benz.atom = c_h
    mol_benz.basis = basis
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz

def build_benz_dist_2(alpha, basis='sto-3g'):
    """
    Input : alpha (varying parameter), basis set
    Ouput : benzene PySCF molecule under distortion 2 (according to paper ...)
    """
    mol_benz = gto.Mole()
    mol_benz.verbose = 5
    mol_benz.output = 'benz.log'
    R1 = 1.39
    R_CH = 1.09
    R2 = 2*R1*np.cos(np.pi/6)*alpha
    x1 = R1*np.sin(np.pi/6)
    X = [0,R1,R1+x1,R1,0,-x1]
    Y = [0,0,R2/2,R2,R2,R2/2]
    X.append(X[0])
    Y.append(Y[0])
    X_H, Y_H = [], []
    xh = R_CH*np.cos(np.pi/3)
    yh = R_CH*np.sin(np.pi/3)
    X_H = [-xh,xh,R_CH,xh,-xh,-R_CH]
    Y_H = [-yh,-yh,0,yh,yh,0]
    c_h = ''
    for i in range(6):
        X_H[i] += X[i]
        Y_H[i] += Y[i]
        c_h += f'C {X[i]} {Y[i]} 0; H {X_H[i]} {Y_H[i]} 0;'
    mol_benz.atom = c_h
    mol_benz.basis = basis
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz

def build_benz_dist_3(alpha, basis='sto-3g'):
    """
    Input : alpha (varying parameter), basis set
    Ouput : benzene PySCF molecule under distortion 3 (according to paper ...)
    """
    mol_benz = gto.Mole()
    mol_benz.verbose = 5
    mol_benz.output = 'benz.log'
    R10 = 1.39
    R_CH = 1.09
    R2 = 2*R10*np.cos(np.pi/6)
    x1 = R10*np.sin(np.pi/6)
    R1 = R10*alpha
    X = [0,R1,R1+x1,R1,0,-x1]
    Y = [0,0,R2/2,R2,R2,R2/2]
    X.append(X[0])
    Y.append(Y[0])
    X_H, Y_H = [], []
    xh = R_CH*np.cos(np.pi/3)
    yh = R_CH*np.sin(np.pi/3)
    X_H = [-xh,xh,R_CH,xh,-xh,-R_CH]
    Y_H = [-yh,-yh,0,yh,yh,0]
    c_h = ''
    for i in range(6):
        X_H[i] += X[i]
        Y_H[i] += Y[i]
        c_h += f'C {X[i]} {Y[i]} 0; H {X_H[i]} {Y_H[i]} 0;'
    mol_benz.atom = c_h
    mol_benz.basis = basis
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz


def full_hamilt_computation(dist, alpha, basis, nb_homo, nb_lumo, calc_E_exact = False):
    """
    Input : 
        - dist : choice of distortion applied to benzene
        - alpha : distorsion parameter
        - basis : basis set
        - nb_homo, nb_lumo : characterizes the active space reduction
        - calc_E_exact : choice of exact energy calculation
    Output : 
        - dictionary containing Hamiltonian of this benzene created by save_H_into_dict 
    """
    if dist==1:
        mol, m_mol = build_benz_dist_1(alpha, basis)
    elif dist==2:
        mol, m_mol = build_benz_dist_2(alpha, basis)
    elif dist==3:
        mol, m_mol = build_benz_dist_3(alpha, basis)
    else:
        print(f'Error : dist = 1, 2 or 3')
    
    save_filename = f'benzene_dist{dist}'
    dic_H_save = save_H_into_dict(alpha, save_filename, mol, m_mol, nb_homo, nb_lumo, calc_E_exact)
    return dic_H_save
    

#################################
##### Graphene + CO2 system #####
#################################

def new_graphene(alpha, nx, ny, **kwargs):
    """
    Input : 
        - alpha  : varying parameter to control carbon-carbon distance in graphene
        - nx, ny : parameters to control graphene sheet size
        - kwargs : th_ext_H  : angle of Hydrogen atom on the final carbon (first layer)
                   th_ext_H2 : angle of Hydrogen atom on the final carbon (second layer)
    Ouput : 
        - c_h    : geometry of graphene sheet in str format (compatible for further PySCF simulations)
    """
    R_CC = 1.39*alpha
    R_CH = 1.01
    try:
        th_ext_H = kwargs["th_ext_H"]
    except:        
        th_ext_H = 30
    try:
        th_ext_H2 = kwargs["th_ext_H2"]
    except:        
        th_ext_H2 = 30
    c_h = ''
    l_1, l_2 = [],[]
    l_1H, l_2H = [], []
    for i_y in range(ny):
        for i_x in range(nx):
            x1, y1 = 2*i_x*np.cos(np.radians(30))*R_CC, (3*i_y+2)*R_CC
            x2, y2 = (2*i_x+1)*np.cos(np.radians(30))*R_CC, (3*i_y+2)*R_CC + np.sin(np.radians(30))*R_CC
            l_1.append((x1,y1))
            l_2.append((x2,y2))
            c_h += f'C {x1} {y1} 0; '
            c_h += f'C {x2} {y2} 0; '
            if i_y == ny-1:
                if i_x == 0:
                    xH1, yH1 = - R_CH*np.cos(np.radians(30)), (3*i_y+2)*R_CC + np.sin(np.radians(30))*R_CC
                    xH2, yH2 = x2, y2 + R_CH
                    l_1H.append( (xH1, yH1) )
                    l_2H.append( (xH2, yH2) )
                    c_h += f'H {xH1} {yH1} 0; '
                    c_h += f'H {xH2} {yH2} 0; '
                else:
                    xH1, yH1 = x2, y2 + R_CH
                    l_1H.append( (xH1, yH1) )
                    c_h += f'H {xH1} {yH1} 0; '
            else:
                if i_x == 0:
                    xH1, yH1 = - R_CH*np.cos(np.radians(th_ext_H)), (3*i_y+2)*R_CC + np.sin(np.radians(th_ext_H))*R_CC
                    l_1H.append( (xH1, yH1) )
                    c_h += f'H {xH1} {yH1} 0; '
            
        x1, y1 = 2*(nx)*np.cos(np.radians(30))*R_CC, (3*i_y+2)*R_CC
        l_1.append( (x1, y1) )
        c_h += f'C {x1} {y1} 0; '
        xH1, yH1 = x1 + R_CH*np.cos(np.radians(th_ext_H)), y1 + R_CH*np.sin(np.radians(th_ext_H))
        l_1H.append( (xH1, yH1) )
        c_h += f'H {xH1} {yH1} 0; '
        
    for i_y in range(ny):
        for i_x in range(nx):
            x1, y1 = 2*i_x*np.cos(np.radians(30))*R_CC, (3*i_y+1)*R_CC
            x2, y2 = (2*i_x+1)*np.cos(np.radians(30))*R_CC, (3*i_y+1)*R_CC - np.sin(np.radians(30))*R_CC
            l_1.append((x1,y1))
            l_2.append((x2,y2))
            c_h += f'C {x1} {y1} 0; '
            c_h += f'C {x2} {y2} 0; '
            
            if i_x == 0:
                xH1, yH1 = x1 - R_CH*np.cos(np.radians(th_ext_H2)), (3*i_y+1)*R_CC - np.sin(np.radians(th_ext_H2))*R_CC
                l_1H.append( (xH1, yH1) )
                c_h += f'H {xH1} {yH1} 0; '
            
        x1, y1 = 2*nx*np.cos(np.radians(30))*R_CC, (3*i_y+1)*R_CC
        l_1.append((x1, y1))
        c_h += f'C {x1} {y1} 0; '
        xH1, yH1 = x1 + R_CH*np.cos(np.radians(th_ext_H2)), y1 - R_CH*np.sin(np.radians(th_ext_H2))
        l_1H.append( (xH1, yH1) )
        c_h += f'H {xH1} {yH1} 0; '
        
    ## ~~ SYMMETRISATION ~~
    for A in l_1+l_2:
        c_h += f'C {A[0]} {-A[1]} 0; '
    for B in l_1H + l_2H:
        c_h += f'H {B[0]} {-B[1]} 0; '
    return c_h


def new_graphene_co2(d_graph_co2, alpha, nx, ny, **kwargs):
    """
    Input : 
        - d_graph_co2 : distance between graphene sheet and CO2 molecule
        - alpha       : varying parameter to control carbon-carbon distance in graphene
        - nx, ny      : parameters to control graphene sheet size
        - kwargs :      th_ext_H  : angle of Hydrogen atom on the final carbon (first layer)
                        th_ext_H2 : angle of Hydrogen atom on the final carbon (second layer) 
                        th_co2    : spherical coordinates of CO2
                        phi_co2   : spherical coordinates of CO2
                        xC        : x coordinate of C atom of CO2
                        yC        : y coordinate of C atom of CO2
                        R_CO      : CO2 interatomic carbon-oxygen distance

    Ouput : 
        - c_h     : geometry of graphene sheet in str format (compatible for further PySCF simulations)
        - str_co2 : geometry of CO2 in str format (compatible for further PySCF simulations)
    """
    ## graphene
    c_h = new_graphene(alpha, nx, ny, **kwargs)
    
    ## co2
    try:
        th_co2 = kwargs["th_co2"]
    except:
        th_co2 = 90
    try:
        phi_co2 = kwargs['phi_co2']
    except:
        phi_co2 = 0
    try:
        xC = kwargs['xC']
    except:
        xC = nx*np.cos(np.radians(30))*alpha*1.39
    try:
        yC = kwargs['yC']
    except:
        yC = 0
    try:
        R_CO_CO2 = kwargs['R_CO']
    except:
        R_CO_CO2 = 1.16
        
    zC = d_graph_co2
    str_co2 = f'C {xC} {yC} {zC}; '
    
    xO1 = xC + R_CO_CO2*np.sin(np.radians(th_co2))*np.cos(np.radians(phi_co2))
    yO1 = yC + R_CO_CO2*np.sin(np.radians(th_co2))*np.sin(np.radians(phi_co2))
    zO1 = zC + R_CO_CO2*np.cos(np.radians(th_co2))
    
    xO2 = xC - R_CO_CO2*np.sin(np.radians(th_co2))*np.cos(np.radians(phi_co2))
    yO2 = yC - R_CO_CO2*np.sin(np.radians(th_co2))*np.sin(np.radians(phi_co2))
    zO2 = zC - R_CO_CO2*np.cos(np.radians(th_co2))
    
    str_co2 += f'O {xO1} {yO1} {zO1}; '
    str_co2 += f'O {xO2} {yO2} {zO2}; '    
    
    return c_h, str_co2


def graphene_co2_dist(d_graph_co2, alpha, nx, ny, dm0=None, **kwargs):
    """
    Input : 
        - d_graph_co2 : distance between graphene sheet and CO2 molecule
        - alpha       : varying parameter to control carbon-carbon distance in graphene
        - nx, ny      : parameters to control graphene sheet size
        - dm0         : initial density matrix for PySCF calculation (default : None)
        - kwargs :      th_ext_H  : angle of Hydrogen atom on the final carbon (first layer)
                        th_ext_H2 : angle of Hydrogen atom on the final carbon (second layer) 
                        th_co2    : spherical coordinates of CO2
                        phi_co2   : spherical coordinates of CO2
                        xC        : x coordinate of C atom of CO2
                        yC        : y coordinate of C atom of CO2
                        R_CO      : CO2 interatomic carbon-oxygen distance
                        basis_set : chemical basis set (default : sto-3g)

    Output :
        - mol, m_mol : PySCF objects containing graphene + CO2 system
    """
    c_h, str_co2 = new_graphene_co2(d_graph_co2=d_graph_co2, alpha=alpha, nx=nx, ny=ny, **kwargs)
    c_gr = c_h + str_co2
    
    mol = gto.Mole()
    mol.atom = c_gr
    
    try:
        basis_set = kwargs['basis_set']
    except:
        basis_set = 'sto-3g'
    mol.basis = basis_set
    mol.max_memory = 100000
    
    mol.spin = 0
    mol.build(0,0)
    m_mol = scf.RHF(mol)
    
    try:
        m_mol.kernel(dm0=dm0)
    except:
        m_mol.kernel()
        
    if not m_mol.converged:
        print(f'/!\ Not converged => Newton method')
        m_mol = scf.RHF(mol).newton()
        m_mol.conv_tol = 1e-6
        m_mol.diis_space = 20
        m_mol.max_cycle = 1500
        m_mol.level_shift = 0.1
        try:
            m_mol.kernel(dm0=dm0)
        except:
            m_mol.kernel()
            
    return mol, m_mol
