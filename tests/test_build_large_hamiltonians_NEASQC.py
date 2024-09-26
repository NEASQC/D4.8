#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pickle

from Github_calc_Hamilt import save_H_into_dict, graphene_co2_dist
    
    
############################################################
##### Prepares the Hamiltonians for Graphene+CO2 calc. #####
############################################################

nx, ny = 4,1
basis_set = 'sto-3g'

hamilt_filename = f'graphene_co2_{nx}_{ny}_{basis_set}_LARGE'
save_filename = f'graphene_co2_{nx}_{ny}_{basis_set}_LARGE'

l_nl = [4]
l_nh = [4]

l_d_graph_co2 = [3.0]                            ## Test

print(f'\n########## MOLECULE : Graphene ({nx},{ny}) + CO2  ##########\n')    

for nb_homo, nb_lumo in zip(l_nh, l_nl):
    print(f'\n----- {nb_homo} HOMO - {nb_lumo} LUMO -----')
       
    for i_co2, d_graph_co2 in enumerate(l_d_graph_co2):
        print(f'- - - - {i_co2+1}/{len(l_d_graph_co2)} - - - -')
        
        ## HOLLOW approach (cf. https://pubs.acs.org/doi/10.1021/acsomega.3c03251)
        mol, m_mol = graphene_co2_dist(d_graph_co2=d_graph_co2, alpha=1, 
                                       nx=nx, ny=ny, 
                                       dm0=None,
                                       th_ext_H=30, th_ext_H2=-5,
                                       th_co2 = 90, phi_co2 = 0,
                                       basis_set=basis_set)

        dic_H_save = save_H_into_dict(d_graph_co2, save_filename, mol, m_mol, nb_homo, nb_lumo, calc_E_exact=False)

