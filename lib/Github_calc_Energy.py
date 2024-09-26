#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle
import scipy.optimize
from pyscf import gto, scf

from qat.lang.AQASM import H, RX, RY, CNOT, QRoutine, Program, X
from qat.fermion.chemistry.ucc_deprecated import get_cluster_ops_and_init_guess
from qat.fermion.chemistry.ucc import tobin

from qat.fermion.transforms import transform_to_jw_basis, get_jw_code, recode_integer
from qat.fermion.trotterisation import make_spin_hamiltonian_trotter_slice

from qat.qpus import LinAlg

#################
### HE Method ###
#################

def HE_circuit_for_ansatz(theta, nbqbits):
    """
    Input : 
        - theta : list of parameters
        - nbqbits : number of qubits of the Hamiltonian
    Output :
        - Qrout : object containing |psi(theta)> obtained with HE method
    """
    
    Qrout = QRoutine()
    d_inter = int(len(theta)/(2*(1+3*(nbqbits-1))))
    LL_inter = np.split(theta,d_inter)
    for ind in range(int(nbqbits)):
        Qrout.apply(H, ind)
    for L in LL_inter:
        for ind in range(nbqbits):
            Qrout.apply(RY(L[ind]),ind)
            Qrout.apply(RX(L[ind+nbqbits]),ind)
        compteur_nq = 2*nbqbits
        for ind in range(nbqbits):
            if ind%nbqbits + 1 < nbqbits:
                Qrout.apply(CNOT, ind, ind+1)
                Qrout.apply(RY(L[ind+compteur_nq]),ind)
                compteur_nq += 1
                Qrout.apply(RX(L[ind+compteur_nq]),ind)
                compteur_nq += 1
                Qrout.apply(RY(L[ind+compteur_nq]),ind+1)
                compteur_nq += 1
                Qrout.apply(RX(L[ind+compteur_nq]),ind+1)
    return Qrout

def fun_HE_ansatz(H_active_sp, theta, nbshots, qpu):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian 
        - theta : list of parameters
        - nbshots : number of shots for quantum measurement
        - qpu : specify the quantum processing unit
    Output :
        - res.value : estimation of <psi(theta)|H_active_sp|psi(theta)>
    """
    
    global compteur
    compteur += 1
    
    prog = Program()
    reg = prog.qalloc(H_active_sp.nbqbits)
    prog.apply(HE_circuit_for_ansatz(theta, H_active_sp.nbqbits), reg)
    circ = prog.to_circ()
    job = circ.to_job(job_type="OBS", observable=H_active_sp, nbshots=nbshots)
    res = qpu.submit(job)
    
    return res.value  

def vqe_he_calc(H_active_sp, d, nbshots):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian 
        - d : depth or number of parameterized layers of the circuit
        - nbshots : number of shots for quantum measurement
    Output :
        - res.fun : minimum of E(theta) = <psi(theta)|H_active_sp|psi(theta)>
    """
    
    qpu = LinAlg()
    depth = d*2*(1+3*(H_active_sp.nbqbits-1))
    theta0 = np.random.random(depth)
    print(f'theta0 = {theta0}')
    
    global compteur
    compteur = 0
    res = scipy.optimize.minimize(lambda theta: fun_HE_ansatz(H_active_sp, theta, nbshots, qpu), theta0, method="COBYLA", options={'maxiter': 1000})
    
    print('--> Ansatz : HE')
    print(f"E (VQE) = {res.fun}")
    print(f'Number of optimization steps : {compteur}')
    print(f'Number of parameters : {len(res.x)}')
    print(f"optimal theta (VQE) = {res.x}")  
    
    return res.fun

############################
##### AJOUT 24/06/2024 #####
############################

def trace_fun_HE_ansatz(H_active_sp, theta, nbshots, qpu):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian 
        - theta : list of parameters
        - nbshots : number of shots for quantum measurement
        - qpu : specify the quantum processing unit
    Output :
        - res.value : estimation of <psi(theta)|H_active_sp|psi(theta)>
    """
    
    global compteur
    compteur += 1
    
    prog = Program()
    reg = prog.qalloc(H_active_sp.nbqbits)
    prog.apply(HE_circuit_for_ansatz(theta, H_active_sp.nbqbits), reg)
    circ = prog.to_circ()
    job = circ.to_job(job_type="OBS", observable=H_active_sp, nbshots=nbshots)
    res = qpu.submit(job)
    
    ## Save energy
    global l_trace_energy
    l_trace_energy.append(res.value)
    
    ## Save theta
    global l_trace_theta
    l_trace_theta.append(theta)

    return res.value  


def trace_vqe_he_calc(H_active_sp, d, nbshots):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian 
        - d : depth or number of parameterized layers of the circuit
        - nbshots : number of shots for quantum measurement
    Output :
        - res.fun : minimum of E(theta) = <psi(theta)|H_active_sp|psi(theta)>
    """
    
    qpu = LinAlg()
    depth = d*2*(1+3*(H_active_sp.nbqbits-1))
    theta0 = np.random.random(depth)
    print(f'theta0 = {theta0}')
    
    global compteur
    compteur = 0
    
    global l_trace_energy
    l_trace_energy = []
    global l_trace_theta
    l_trace_theta = []
    
    res = scipy.optimize.minimize(lambda theta: trace_fun_HE_ansatz(H_active_sp, theta, nbshots, qpu), 
                                  theta0, 
                                  method="COBYLA", options={'maxiter': 5e3})
    
    print('--> Ansatz : HE')
    print(f"E (VQE) = {res.fun}")
    print(f'Number of optimization steps : {compteur}')
    print(f'Number of parameters : {len(res.x)}')
    print(f"optimal theta (VQE) = {res.x}")  
    
    return res.fun

############################
############################

###################
### qUCC Method ###
###################

def ucc_ansatz_calc(H_active, active_inds, occ_inds, noons, orbital_energies, nels):
    """
    Input : 
        - H_active : Hamiltonian after active space reduction (orbital freezing)
        - active_inds, occ_inds :  list of index of active/occupied orbitals
        - noons : list of natural-orbital occupation numbers 
        - orbital energies : list of energies of each molecular orbital
        - nels : total number of electrons
    Output : 
        - H_active_sp : Jordan-Wigner Hamiltonian
        - qprog : quantum circuit of |psi> obtained with qUCC method
        - theta_0 : initial guess of parameters of |psi> 
    """
    
    ########
    # 1 : preparation
    ########
    
    active_noons, active_orb_energies = [], []
    for ind in active_inds:
        active_noons.extend([noons[ind], noons[ind]])
        active_orb_energies.extend([orbital_energies[ind], orbital_energies[ind]])

    nb_active_els = nels - 2*len(occ_inds)
    cluster_ops, theta_0, hf_init = get_cluster_ops_and_init_guess(nb_active_els,
                                                                active_noons,
                                                                active_orb_energies,
                                                                H_active.hpqrs)
    
    ########
    # 2 : transformation to Jordan-Wigner basis
    ########
    
    H_active_sp = transform_to_jw_basis(H_active)
    cluster_ops_sp = [transform_to_jw_basis(t_o) for t_o in cluster_ops]
    hf_init_sp = recode_integer(hf_init, get_jw_code(H_active_sp.nbqbits))
    
    ########
    # 4 : creation of the qUCC ansatz
    ########
    
    def build_qprog(theta):
    
        ## Preparation of |HF> ##
        ket_hf_init_sp = [int(c) for c in tobin(hf_init_sp, H_active_sp.nbqbits)]
        QPROG_exp = QRoutine(arity=H_active_sp.nbqbits)
        for j in range(H_active_sp.nbqbits):
            if int(ket_hf_init_sp[j]) == 1:
                QPROG_exp.apply(X, j)

        ## Implement \prod_{i} exp(A_i*theta_i) to |HF> ##
        reg2 = list(range(H_active_sp.nbqbits))
        for n_term, (term, theta_term) in enumerate(zip(cluster_ops_sp, theta)):
            if type(theta_term) == list:
                theta_term = theta_term[0]
            op_exp = make_spin_hamiltonian_trotter_slice(term, coeff=float(theta_term))
            QPROG_exp.apply(op_exp, reg2[: op_exp.arity])
        return QPROG_exp
    
    qprog = build_qprog

    return H_active_sp, qprog, theta_0


def fun_qucc_ansatz(H_active_sp, qrout, theta, nbshots):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian
        - qrout : quantum circuit of |psi> obtained with qUCC method
        - theta : list of parameters
        - nbshots : number of shots for quantum measurement
    Output : 
        - res.value : estimation of <psi(theta)|H_active_sp|psi(theta)>        
    """
    global compteur
    compteur += 1
    
    qpu = LinAlg()
    prog = Program()
    reg = prog.qalloc(H_active_sp.nbqbits)
    prog.apply(qrout(theta), reg)
    circ = prog.to_circ()
    job = circ.to_job(job_type="OBS", observable=H_active_sp, nbshots=nbshots)
    res = qpu.submit(job)
    return res.value

def vqe_ucc_calc(H_active_sp, qprog, theta_0, nbshots=0):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian
        - qprog : quantum circuit of |psi> obtained with qUCC method
        - theta_0 : initial guess of parameters of |psi> 
    Output :
        - res.fun : minimum of E(theta) = <psi(theta)|H_active_sp|psi(theta)>    
    """
    
    global compteur
    compteur = 0
    
    res = scipy.optimize.minimize(lambda theta: fun_qucc_ansatz(H_active_sp, qprog, theta, nbshots), theta_0, method= "COBYLA", options={'maxiter': 1000})
    
    print('--> Ansatz : UCC')
    print(f"E (VQE) = {res.fun}")
    print(f'Number of optimization steps : {compteur}')
    print(f'Number of parameters : {len(res.x)}')
    print(f"optimal theta (VQE) = {res.x}")
    
    return res.fun


############################
##### AJOUT 24/06/2024 #####
############################

def trace_fun_qucc_ansatz(H_active_sp, qrout, theta, nbshots, str_key_complete):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian
        - qrout : quantum circuit of |psi> obtained with qUCC method
        - theta : list of parameters
        - nbshots : number of shots for quantum measurement
    Output : 
        - res.value : estimation of <psi(theta)|H_active_sp|psi(theta)>        
    """
    global compteur
    compteur += 1
    
    qpu = LinAlg()
    prog = Program()
    reg = prog.qalloc(H_active_sp.nbqbits)
    prog.apply(qrout(theta), reg)
    circ = prog.to_circ()
    job = circ.to_job(job_type="OBS", observable=H_active_sp, nbshots=nbshots)
    res = qpu.submit(job)
    
    ## Save energy
    global l_trace_energy
    l_trace_energy.append(res.value)
    
    ## Save theta
    global l_trace_theta
    l_trace_theta.append(theta)
    
    str_key_complete = '#'.join(str_key_complete)
    
    try:
        with open(f'temporary_results.E.pickle','rb') as f1:
            dic_temp = pickle.load(f1)
    except:
        dic_temp = {}
    try:
        dic_temp[str_key_complete]
    except:
        dic_temp[str_key_complete] = {'l_E':[[res.value]], 'l_theta':[[theta]]}
    else:
        dic_temp[str_key_complete]['l_E'][-1].append(res.value)
        dic_temp[str_key_complete]['l_theta'][-1].append(theta)
    
    with open(f'temporary_results.E.pickle','wb') as f1:
        pickle.dump(dic_temp, f1)
    
    return res.value

def trace_vqe_ucc_calc(H_active_sp, qprog, theta_0, nbshots=0, str_key_complete=''):
    """
    Input : 
        - H_active_sp : Jordan-Wigner Hamiltonian
        - qprog : quantum circuit of |psi> obtained with qUCC method
        - theta_0 : initial guess of parameters of |psi> 
    Output :
        - res.fun : minimum of E(theta) = <psi(theta)|H_active_sp|psi(theta)>    
    """
    
    global compteur
    compteur = 0
    
    global l_trace_energy
    l_trace_energy = []
    global l_trace_theta
    l_trace_theta = []
    
    res = scipy.optimize.minimize(lambda theta: trace_fun_qucc_ansatz(H_active_sp, qprog, theta, nbshots, str_key_complete), 
                                  theta_0, 
                                  method= "COBYLA", 
                                  tol=1e-5,
                                  options={'maxiter': 1.5e4})
    
    print('--> Ansatz : UCC')
    print(f"E (VQE) = {res.fun}")
    print(f'Number of optimization steps : {compteur}')
    print(f'Number of parameters : {len(res.x)}')
    print(f"optimal theta (VQE) = {res.x}")
    
    return res.fun

############################
############################

####################
### Results save ###
####################

def save_E_into_dict(l1, hamilt_filename, save_filename, mol, m_mol, nb_homo, nb_lumo, ansatz="qUCC", nbshots=0, d=1, N_trials=1):
    """
    Input : 
        - l1 : varying parameter
        - hamilt_filename : filename that contains the Hamiltonian 
        - save_filename : filename for saving
        - mol, m_mol : PySCF molecule
        - nb_homo, nb_lumo : characterizes the active space reduction
        - ansatz : choice of method to create |psi> (qUCC or HE)
        - nbshots : number of shots for quantum measurement
        - d : depth or number of parameterized layers of the circuit (only for HE)
        - N_trials : number of times one wants to repeat the calculation of E_vqe
    Output :
        - dic_E_save : dictionary contained in save_filename.E.pickle with
               * 1st key : varying parameter (e.g. : bond length of a molecule)
               * 2nd key : chemical basis set
               * 3rd key : nb_homo (characterizes the active space reduction)
               * 4th key : nb_lumo (characterizes the active space reduction)
               * Then, on the same level : 
                   --> HF (Hartree Fock energy) : unique value
                   --> VQE (for VQE energies)
                        ↳ qUCC => nbshots => list with N_trials VQE energies
                        ↳ HE => nbshots => d => list with N_trials VQE energies
    """
    with open(f'{hamilt_filename}','rb') as f0:
        dic_H_save = pickle.load(f0)
    try:
        with open(f'{save_filename}.E.pickle','rb') as f1:
            dic_E_save = pickle.load(f1)
    except:
        print(f'Error : the dictionary {save_filename}.E.pickle doesn\'t exist.\n => Creation of a new one')
        dic_E_save = {}
    
    try:
        with open(f'{save_filename}.E.trace.pickle','rb') as f1:
            dic_E_save_trace = pickle.load(f1)
    except:
        print(f'Error : the dictionary {save_filename}.E.trace.pickle doesn\'t exist.\n => Creation of a new one')
        dic_E_save_trace = {}
        
    H_active = dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['H_active']
    active_inds = dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['active_inds']
    occ_inds = dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['occ_inds']
    noons = dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['noons']
    orbital_energies = dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['orbital_energies']
    nels = dic_H_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['nels']
    
    try:
        dic_E_save[str(l1)]
    except:
        dic_E_save[str(l1)] = {}
    try:
        dic_E_save[str(l1)][mol.basis]
    except:
        dic_E_save[str(l1)][mol.basis] = {}
    try:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)]
    except:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)] = {}
    try:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]
    except:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)] = {}
    try:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['HF']
    except:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['HF'] = m_mol.e_tot
    
    try:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']
    except:
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE'] = {}
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'] = {}
        dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'] = {}
    
    try:
        dic_E_save_trace[str(l1)]
    except:
        dic_E_save_trace[str(l1)] = {}
    try:
        dic_E_save_trace[str(l1)][mol.basis]
    except:
        dic_E_save_trace[str(l1)][mol.basis] = {}
    try:
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)]
    except:
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)] = {}
    try:
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]
    except:
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)] = {}   
    try:
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']
    except:
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE'] = {}
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'] = {}
        dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'] = {}
        
    if ansatz == "qUCC":
        H_active_sp, qprog, theta_0 = ucc_ansatz_calc(H_active, active_inds, occ_inds, noons, orbital_energies, nels)
        
        try:
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]
        except KeyError:
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)] = []
        
        try:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]
        except KeyError:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)] = {}
        try:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]['l_E']
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]['l_theta']
        except KeyError:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]['l_E'] = []
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]['l_theta'] = []
        
        str_key_complete = [str(l1), mol.basis, str(nb_homo), str(nb_lumo), 'VQE', 'qUCC', str(nbshots)]
        for _ in range(N_trials):
            print(f'############ Trial : {_+1}/{N_trials} ############')
            
            E_vqe = trace_vqe_ucc_calc(H_active_sp, qprog, theta_0, nbshots, str_key_complete)
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)].append(E_vqe)
            print(f'\n')
            with open(f'{save_filename}.E.pickle','wb') as f2:
                pickle.dump(dic_E_save,f2)
                
            global l_trace_theta
            global l_trace_energy
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]['l_E'].append(l_trace_energy)
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['qUCC'][str(nbshots)]['l_theta'].append(l_trace_theta)
            with open(f'{save_filename}.E.trace.pickle','wb') as f2:
                pickle.dump(dic_E_save_trace,f2)
            
    elif ansatz == 'HE':
        H_active_sp = transform_to_jw_basis(H_active)
        
        try:
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)]
        except KeyError:
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)] = {}
        try:
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]
        except KeyError:
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)] = []
        
        try:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)]
        except KeyError:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)] = {}
        try:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]
        except KeyError:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)] = {}
        try:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]['l_E']
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]['l_theta']
        except KeyError:
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]['l_E'] = []
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]['l_theta'] = []
        
        
        for _ in range(N_trials):
            print(f'############ Trial : {_+1}/{N_trials} ############')
            
            E_vqe = trace_vqe_he_calc(H_active_sp, d, nbshots)
            dic_E_save[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)].append(E_vqe)
            print(f'\n')
            with open(f'{save_filename}.E.pickle','wb') as f2:
                pickle.dump(dic_E_save,f2)
#             global l_trace_theta
#             global l_trace_energy
            
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]['l_E'].append(l_trace_energy)
            dic_E_save_trace[str(l1)][mol.basis][str(nb_homo)][str(nb_lumo)]['VQE']['HE'][str(nbshots)][str(d)]['l_theta'].append(l_trace_theta)
            with open(f'{save_filename}.E.trace.pickle','wb') as f2:
                pickle.dump(dic_E_save_trace,f2)
                
    else:
        print(f'Error : ansatz must be "qUCC" or "HE')
        
    with open(f'{save_filename}.E.pickle','wb') as f2:
        pickle.dump(dic_E_save,f2)
        print(f'=> The dictionary of results is saved in {save_filename}.E.pickle')
        
    with open(f'{save_filename}.E.trace.pickle','wb') as f2:
        pickle.dump(dic_E_save_trace,f2)
        print(f'=> The dictionary of traced results is saved in {save_filename}.E.trace.pickle')
    return dic_E_save



#######################
### Results display ###
#######################

def display_results(testeur_E):
    for cle in testeur_E:
        for cle2 in testeur_E[cle]:
            print(f'{cle} HOMO - {cle2} LUMO :')
            for cle3 in testeur_E[cle][cle2]:
                if cle3 == 'HF':
                    if bool(testeur_E[cle][cle2][cle3]) == True:
                        print(f'      {cle3} : calculated')
                    else:
                        print(f'      {cle3} : ---')
                else:
                    str_cle3 = f'      {cle3} :'
                    for cle4 in testeur_E[cle][cle2][cle3]:
                        if cle4 == 'qUCC':
                            str_cle3 += f' {cle4} :'                                 
                            if bool(testeur_E[cle][cle2][cle3][cle4]) == False:
                                str_cle3 += f' ---'
                                print(str_cle3)
                            else:                                    
                                L_nbshots = list(testeur_E[cle][cle2][cle3][cle4].keys())
                                L_len = [len(testeur_E[cle][cle2][cle3][cle4][cle5]) for cle5 in L_nbshots]
                                nb0 = L_nbshots.pop(0)
                                len0 = L_len.pop(0)
                                str_cle3 += f' nbshots = {nb0} : ({len0} trials)'
                                print(str_cle3)
                                for nbs in L_nbshots:
                                    str_0 = f'                 '
                                    str_0 += f' nbshots = {nbs} : ({L_len[L_nbshots.index(nbs)]} trials)'
                                    print(str_0)
                        else:
                            str_cle4 = f'            {cle4} :'
                            if bool(testeur_E[cle][cle2][cle3][cle4]) == False:
                                str_cle4 += f' ---'
                                print(str_cle4)
                            else:
                                L_nbshots = list(testeur_E[cle][cle2][cle3][cle4].keys())
                                nb0 = L_nbshots.pop(0)
                                str_cle4 += f' nbshots = {nb0} :'
                                L_d0 = list(testeur_E[cle][cle2][cle3][cle4][nb0].keys())
                                L_len0 = [len(testeur_E[cle][cle2][cle3][cle4][nb0][cle6]) for cle6 in L_d0]
                                try:
                                    str_cle4 += f' d = {L_d0.pop(0)} ({L_len0.pop(0)} trials)'
                                except:
                                    str_cle4 += f' --- '
                                print(str_cle4)
                                for _ in range(len(L_d0)):
                                    str_0 = f'                              '
                                    str_0 += f' d = {L_d0[_]} ({L_len0[_]} trials)'
                                    print(str_0)
                                for nbs in L_nbshots:
                                    str_nbs = f'                '
                                    str_nbs += f' nbshots = {nbs} :'
                                    L_d = list(testeur_E[cle][cle2][cle3][cle4][str(nbs)].keys())
                                    L_len = [len(testeur_E[cle][cle2][cle3][cle4][str(nbs)][cle6]) for cle6 in L_d]
                                    try:
                                        str_nbs += f' d = {L_d.pop(0)} ({L_len.pop(0)} trials)'
                                    except:
                                        str_nbs += f' --- '
                                    print(str_nbs)
                                    for _ in range(len(L_d)):
                                        str_d = f'                              '
                                        str_d += f' d = {L_d[_]} ({L_len[_]} trials)'
                                        print(str_d)
        print('================================================')

def display_full_dic(dic_E):
    for cle in dic_E:
        print(f'####################{(len(cle)-1)*"#"}')
        print(f'###### l1 = {cle} ######')
        print(f'####################{(len(cle)-1)*"#"}')
        for cle2 in dic_E[cle]:
            print(f'--> basis : {cle2}')
            display_results(dic_E[cle][cle2])      
        
##############################
### Distortions of benzene ###
##############################

def build_benz_dist_1(alpha):
    """
    Input : alpha
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
    mol_benz.basis = 'sto-3g'
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz

def build_benz_dist_2(alpha):
    """
    Input : alpha
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
    mol_benz.basis = 'sto-3g'
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz

def build_benz_dist_3(alpha):
    """
    Input : alpha
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
    mol_benz.basis = 'sto-3g'
    mol_benz.spin = 0
    mol_benz.build(0,0)
    m_benz = scf.RHF(mol_benz)
    m_benz.kernel()
    return mol_benz, m_benz


def full_energy_computation(dist, alpha, basis, nb_homo, nb_lumo, ansatz="qUCC", nbshots=0, d=1, N_trials=1):
    """
    Input : 
        - dist : choice of distortion applied to benzene
        - alpha : distorsion parameter
        - basis : basis set
        - nb_homo, nb_lumo : characterizes the active space reduction
        - ansatz : choice of method to create |psi> (qUCC or HE)
        - nbshots : number of shots for quantum measurement
        - d : depth or number of parameterized layers of the circuit (only for HE)
        - N_trials : number of times one wants to repeat the calculation of E_vqe
    Output : 
        - dictionary containing energies of this benzene created by save_E_into_dict 
    """
    if dist==1:
        mol, m_mol = build_benz_dist_1(alpha, basis)
    elif dist==2:
        mol, m_mol = build_benz_dist_2(alpha, basis)
    elif dist==3:
        mol, m_mol = build_benz_dist_3(alpha, basis)
    else:
        print(f'Error : dist = 1, 2 or 3')
    
    hamilt_filename = f'benzene_dist{dist}.H.pickle'
    save_filename = f'benzene_dist{dist}'
    
    dic_E_save = save_E_into_dict(alpha, hamilt_filename, save_filename, mol, m_mol, nb_homo, nb_lumo, ansatz, nbshots, d, N_trials)
    return dic_E_save
