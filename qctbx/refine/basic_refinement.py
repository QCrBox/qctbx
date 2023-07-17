import numpy as np
from copy import deepcopy
import smtbx
from smtbx.refinement import least_squares
from .normal_eqns import build_refinement_wrapper, normal_eqns
from cctbx.xray.structure import structure 
from cctbx import miller
from ..F0jSourceBase import F0jSource
from typing import Dict, Union
from .normal_eqns import normal_eqns
from smtbx.refinement import constraints


default_har_convergence_conditions = {
    'position(abs)': 1e-4,
    'position/esd' : 1e-2,
    'uij(abs)': 1e-4,
    'uij/esd': 1e-1,
    'max(cycles)': 10
}

def check_convergence_har(
    xray_structure: structure,
    xray_structure0: structure, 
    norm_eq: normal_eqns, 
    har_convergence_conditions: Dict[str, Union[float, int]]
):
    cov_annot = norm_eq.covariance_matrix_and_annotations()
    distances = xray_structure0.distances(xray_structure)
    max_dist_abs = np.max(distances)
    max_dist_ov_esd = -99999.9
    max_uij_ov_esd = -99999.0
    max_uij_abs = -99999.0
    max_dist_atom = xray_structure.scatterers()[np.argmax(distances)].label
    max_dist_esd_atom = ''
    max_uij_atom = ''
    max_uij_esd_atom = ''
    
    for sc, sc0 in zip(xray_structure.scatterers(), xray_structure0.scatterers()):
        for coord_index, coordinate in enumerate(('x', 'y', 'z')):
            coord_label = f'{sc.label}.{coordinate}'
            if coord_label in cov_annot.annotations:
                esd = np.sqrt(cov_annot.variance_of(coord_label))
                delta_ov_esd = np.abs(sc.site[coord_index] - sc0.site[coord_index]) / esd
                if delta_ov_esd > max_dist_ov_esd:
                    max_dist_ov_esd = delta_ov_esd
                    max_dist_esd_atom = sc.label
        if sc.flags.use_u_aniso():
            for uij_index, uij in enumerate(('u11', 'u22', 'u33', 'u12', 'u13', 'u23')):
                uij_label = f'{sc.label}.{uij}'
                if uij_label in cov_annot.annotations:
                    esd = np.sqrt(cov_annot.variance_of(uij_label))
                    delta = np.abs(sc.u_star[uij_index] - sc0.u_star[uij_index])
                    if delta > max_uij_abs:
                        max_uij_abs = delta
                        max_uij_atom = sc.label
                    delta_ov_esd = delta / esd
                    if delta_ov_esd > max_uij_ov_esd:
                        max_uij_ov_esd = delta_ov_esd
                        max_uij_esd_atom = sc.label
        if sc.flags.use_u_iso():
            uij_label = f'{sc.label}.uiso'
            if uij_label in cov_annot.annotations:
                esd = np.sqrt(cov_annot.variance_of(uij_label))
                delta = np.abs(sc.u_iso - sc0.u_iso)
                if delta > max_uij_abs:
                    max_uij_abs = delta
                    max_uij_atom = sc.label
                delta_ov_esd = delta / esd
                if delta_ov_esd > max_uij_ov_esd:
                    max_uij_ov_esd = delta_ov_esd
                    max_uij_esd_atom = sc.label
    conditions = [True] * 4
    if max_dist_abs > har_convergence_conditions['position(abs)']:
        print(f"Positions have not converged (max:{max_dist_abs:.6f}, {max_dist_atom}, criterion:<{har_convergence_conditions['position(abs)']}")
        conditions[0] = False
    if max_dist_ov_esd > har_convergence_conditions['position/esd']:
        print(f"Positions/esd have not converged (max:{max_dist_ov_esd:.6f}, {max_dist_esd_atom}, criterion:<{har_convergence_conditions['position/esd']}")
        conditions[1] = False
    if max_uij_abs > har_convergence_conditions['uij(abs)']:
        print(f"Displacement parameters have not converged (max:{max_uij_abs:.6f}, {max_uij_atom}, criterion:<{har_convergence_conditions['uij(abs)']}")
        conditions[2] = False
    if max_uij_ov_esd > har_convergence_conditions['uij/esd']:
        print(f"Displacement parameters/esd have not converged (max:{max_uij_ov_esd:.6f}, {max_uij_esd_atom}, criterion:<{har_convergence_conditions['uij/esd']}")
        conditions[3] = False

    return all(conditions)


def basic_refinement(
    xray_structure: structure,
    miller_array: miller.array,
    f0jeval: F0jSource,
    har_convergence_conditions: Dict[str, Union[float, int]] = default_har_convergence_conditions,
    constraints_list=[],
    restraints_manager=None,
    solver_name='Gauss-Newton',
    tsc_path='qctbx.tsc'
):

    for i in range(har_convergence_conditions['max(cycles)']):
        xray_structure0 = deepcopy(xray_structure)
        f0jeval.cctbx2tsc(xray_structure, miller_array, tsc_path)

        reparametrisation = constraints.reparametrisation(
            xray_structure, 
            constraints_list,
            smtbx.utils.connectivity_table(xray_structure)
        )
        
        ls = least_squares.crystallographic_ls(
            miller_array.as_xray_observations(),
            reparametrisation,
            non_linear_ls_with_separable_scale_factor=least_squares.normal_eqns.non_linear_ls_with_separable_scale_factor_BLAS_2
        )
        
        norm_eq = normal_eqns(
            miller_array.as_xray_observations(),
            ls,
            table_file_name=tsc_path
        )
        norm_eq.restraints_manager = restraints_manager
        RefinementWrapper =  build_refinement_wrapper(solver_name)
        rfn_wrapper = RefinementWrapper(norm_eq)
        print('-' * 20)
        if check_convergence_har(xray_structure, xray_structure0, norm_eq, har_convergence_conditions):
            break
        print('-' * 20)
    return structure, norm_eq