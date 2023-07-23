"""This module has the building blocks for a basic refinement. The main part is
basically copied from Olex2 with only omissions and tiny modifications.
Currently, it only works with the cctbx fork from https://github.com/pcxod/cctbx_project"""

from scitbx.array_family import flex
from smtbx.refinement import least_squares
from smtbx_refinement_least_squares_ext import (
    f_calc_function_default, MaskData, build_design_matrix
)
from scitbx.lstbx import normal_eqns_solving

from smtbx.structure_factors import direct

import warnings
import sys
import time
import logging


class normal_eqns(least_squares.crystallographic_ls_class()):
    log = None
    std_observations = None

    def __init__(self, observations, refinement, table_file_name=None, **kwds):
        super(normal_eqns, self).__init__(
            observations, refinement.reparametrisation, initial_scale_factor=1.0,
            **kwds
        )
        if table_file_name:
            try:
                one_h_linearisation = direct.f_calc_modulus_squared(
                  self.xray_structure, table_file_name=table_file_name)
            except Exception as e:
                e_str = str(e)
                if "stoks.size() == scatterer" in e_str:
                    print("Number of atoms in model and table are not matching!")
                elif "Error during building of normal equations using OpenMP" in e_str:
                    print("OpenMP Error during Normal Equation build-up, likely missing reflection in .tsc file")
                raise e
        else:
            one_h_linearisation = direct.f_calc_modulus_squared(
                self.xray_structure, reflections=self.observations
            )
        self.refinement = refinement
        self.one_h_linearisation = f_calc_function_default(one_h_linearisation)
        self.f_mask_data = None
        if self.f_mask is None:
            self.f_mask_data = MaskData(flex.complex_double())
        else:
            self.f_mask_data = MaskData(self.observations,
                                        self.xray_structure.space_group(),
                                        self.observations.fo_sq.anomalous_flag(),
                                        self.f_mask.indices(),
                                        self.f_mask.data())
        self.n_current_cycle = 0

    def get_std_DM(self):
        cl = least_squares.crystallographic_ls_class()
        self.reparametrisation.linearise()
        xx = cl(self.observations, self.reparametrisation)
        def args():
            args = (xx,
                self.std_observations,
                self.f_mask_data,
                self.weighting_scheme,
                1.0,
                self.one_h_linearisation,
                self.reparametrisation.jacobian_transpose_matching_grad_fc(),
                self.reparametrisation.fc_correction, False, True, False)
            return args
        self.data = build_design_matrix(*args()) #lock destruction
        return self.data

    def build_up(self, objective_only):
        super(normal_eqns, self).build_up(objective_only)
        return

    def step_forward(self):
        self.n_current_cycle += 1
        self.xray_structure_pre_cycle = self.xray_structure.deep_copy_scatterers()
        super(normal_eqns, self).step_forward()
        self.show_cycle_summary(log=self.log)
        self.show_sorted_shifts(max_items=10, log=self.log)
        self.restraints_manager.show_sorted(
        self.xray_structure, f=self.log)
        self.show_cycle_summary()
        return self

    def step_backward(self):
        super(normal_eqns, self).step_backward()
        return self

    def analyse_shifts(self):
        self.max_shift_esd = None
        self.max_shift_esd_item = None
        self.mean_shift_esd = None
        try:
            self.iterations_object.analyse_shifts()
            self.mean_shift_esd = self.iterations_object.mean_ls_shift_over_su
            self.max_shift_esd = self.iterations_object.max_ls_shift_over_su
            self.max_shift_esd_item = self.iterations_object.max_shift_for
        except Exception as s:
            print(s)

    def show_cycle_summary(self, log=None):
        if log is None: log = sys.stdout
        # self.reparametrisation.n_independents + OSF
        max_shift_site = self.max_shift_site()
        max_shift_u = self.max_shift_u()
        self.analyse_shifts()
        print_tabular = True

        if print_tabular:
            header = "  % 5i    % 6.2f    % 6.2f    % 6.2f    % 8.3f %-11s  % 8.2e %-11s  % 8.2e %-11s"
            params = (self.n_current_cycle,
                self.r1_factor(cutoff_factor=2)[0] * 100,
                self.wR2() * 100,
                self.goof(),
                self.max_shift_esd,
                '(' + self.max_shift_esd_item + ')',
                max_shift_site[0],
                '(' + max_shift_site[1].label + ')',
                max_shift_u[0],
                '(' + max_shift_u[1].label + ')',
            )
            if hasattr(self.iterations_object,'mu'):
                header += "  % 8.2e"
                params += (self.iterations_object.mu,)
            print(header %params, file=log)

        else:
            print("wR2 = %.4f | GooF = %.4f for %i data and %i parameters" %(
                self.wR2(),
                self.goof(),
                self.observations.fo_sq.size(),
                self.reparametrisation.n_independents + 1,
            ), file=log)

            print("Max shifts: ", end=' ', file=log)

            print("Site: %.4f A for %s |" %(
                max_shift_site[0],
                max_shift_site[1].label
            ), end=' ', file=log)
            print("dU = %.4f for %s" %(
                max_shift_u[0],
                max_shift_u[1].label,
            ), file=log)


    def max_shift_site(self):
        return next(self.iter_shifts_sites(max_items=1))

    def max_shift_u(self):
        return next(self.iter_shifts_u(max_items=1))

    def max_shift_esd(self):
        self.get_shifts()
        return next(self.iter_shifts_u(max_items=1))


    def iter_shifts_sites(self, max_items=None):
        scatterers = self.xray_structure.scatterers()
        sites_shifts = (
            self.xray_structure.sites_cart() - self.xray_structure_pre_cycle.sites_cart()
        )
        distances = sites_shifts.norms()
        i_distances_sorted = flex.sort_permutation(data=distances, reverse=True)
        if max_items is not None:
            i_distances_sorted = i_distances_sorted[:max_items]
        for i_seq in iter(i_distances_sorted):
            yield distances[i_seq], scatterers[i_seq]

    def iter_shifts_u(self, max_items=None):
        scatterers = self.xray_structure.scatterers()
        adp_shifts = (
            self.xray_structure.extract_u_cart_plus_u_iso()
            - self.xray_structure_pre_cycle.extract_u_cart_plus_u_iso()
        )
        norms = adp_shifts.norms()
        i_adp_shifts_sorted = flex.sort_permutation(data=norms, reverse=True)
        if max_items is not None:
            i_adp_shifts_sorted = i_adp_shifts_sorted[:max_items]
        for i_seq in iter(i_adp_shifts_sorted):
            yield norms[i_seq], scatterers[i_seq]

    def show_log(self, f=None):
        import sys
        if self.log is sys.stdout:
            return
        if f is None:
            f = sys.stdout
        print(self.log.getvalue(), file=f)

    def show_sorted_shifts(self, max_items=None, log=None):
        import sys
        if log is None:
            log = sys.stdout
        print("Sorted site shifts in Angstrom:", file=log)
        print("shift scatterer", file=log)
        n_not_shown = self.xray_structure.scatterers().size()
        for distance, scatterer in self.iter_shifts_sites(max_items=max_items):
            n_not_shown -= 1
            print("%5.3f %s" %(distance, scatterer.label), file=log)
            if round(distance, 3) == 0:
                break
        if n_not_shown != 0:
            print("... (remaining %d not shown)" % n_not_shown, file=log)
        #
        print("Sorted adp shift norms:", file=log)
        print("dU scatterer", file=log)
        n_not_shown = self.xray_structure.scatterers().size()
        for norm, scatterer in self.iter_shifts_u(max_items=max_items):
            n_not_shown -= 1
            print("%5.3f %s" %(norm, scatterer.label), file=log)
            if round(norm, 3) == 0:
                break
        if n_not_shown != 0:
            print("... (remaining %d not shown)" % n_not_shown, file=log)

    def show_shifts(self, log=None):
        import sys
        if log is None:
            log = sys.stdout
        site_symmetry_table = self.xray_structure.site_symmetry_table()
        i=0
        for i_sc, sc in enumerate(self.xray_structure.scatterers()):
            op = site_symmetry_table.get(i_sc)
            print("%-4s" % sc.label, file=log)
            if sc.flags.grad_site():
                n = op.site_constraints().n_independent_params()
                if n != 0:
                    print(
                        ("site:" + "%7.4f, "*(n-1) + "%7.4f") % tuple(self.shifts[-1][i:i+n]),
                        file=log
                    )
                i += n
            if sc.flags.grad_u_iso() and sc.flags.use_u_iso():
                if not(sc.flags.tan_u_iso() and sc.flags.param > 0):
                  print("u_iso: %6.4f" % self.shifts[i], file=log)
                  i += 1
            if sc.flags.grad_u_aniso() and sc.flags.use_u_aniso():
                n = op.adp_constraints().n_independent_params()
                print(
                    (("u_aniso:" + "%6.3f, "*(n-1) + "%6.3f") % tuple(self.shifts[-1][i:i+n])),
                    file=log
                )
                i += n
            if sc.flags.grad_occupancy():
                print("occ: %4.2f" % self.shifts[-1][i], file=log)
                i += 1
            if sc.flags.grad_fp():
                print("f': %6.4f" % self.shifts[-1][i], file=log)
                i += 1
            if sc.flags.grad_fdp():
                print("f'': %6.4f" % self.shifts[-1][i], file=log)
                i += 1
            print(file=log)


class iterations_with_shift_analysis(normal_eqns_solving.iterations):
    convergence_as_shift_over_esd = 1e-3
    max_ls_shift_over_su = None

    def analyse_shifts(self, limit_shift_over_su=None):
        if self.max_ls_shift_over_su is not None:
            return
        x = self.non_linear_ls.step()
        esd = self.non_linear_ls.covariance_matrix().matrix_packed_u_diagonal()
        ls_shifts_over_su = x/flex.sqrt(esd)
        max_shift_i = 0
        s_sum = 0
        for i,s in enumerate(ls_shifts_over_su):
            s_sum += abs(s)
            if abs(ls_shifts_over_su[max_shift_i]) < abs(s):
                max_shift_i = i
        #max shift for the LS
        self.mean_ls_shift_over_su = s_sum/len(ls_shifts_over_su)
        self.max_ls_shift_over_su = ls_shifts_over_su[max_shift_i]
        r = self.non_linear_ls.actual.reparametrisation
        J = r.jacobian_transpose_matching_grad_fc().transpose()
        spc = 0
        for ip in r.independent_scalar_parameters:
            spc += ip.n_param
        if max_shift_i >= J.n_cols-spc:
            if r.thickness is not None and r.thickness.grad_index == max_shift_i:
                self.max_shift_for = "EDT"
            elif r.fc_correction and r.fc_correction.grad and\
                     max_shift_i >= r.fc_correction.grad_index and\
                     max_shift_i < r.fc_correction.grad_index + r.fc_correction.n_param:
                self.max_shift_for = "EXTI/SWAT"
            else:
                self.max_shift_for = "BASF%s" %(max_shift_i - (J.n_cols - spc) + 1)
        else:
            for e in J.col(max_shift_i):
                self.max_shift_for = r.component_annotations[e[0]]
                break
        #self.shifts_over_su = J * self.ls_shifts_over_su
        #self.max_shift_over_su = flex.max(self.shifts_over_su)
        if abs(self.max_ls_shift_over_su) < self.convergence_as_shift_over_esd:
            return True
        if limit_shift_over_su is not None and abs(self.max_ls_shift_over_su) > limit_shift_over_su:
            shift_scale = limit_shift_over_su/self.max_ls_shift_over_su
            x *= abs(shift_scale)
        return False

    def reset_shifts(self):
        self.max_ls_shift_over_su = None
        self.max_shift_for = None

class naive_iterations_with_damping_and_shift_limit(
        normal_eqns_solving.naive_iterations_with_damping,
        iterations_with_shift_analysis):
    max_shift_over_esd = 15

    def do(self):
        self.n_iterations = 0
        start_t = time.time()
        while self.n_iterations < self.n_max_iterations:
            t1 = time.time()
            self.reset_shifts()
            self.non_linear_ls.actual.xray_structure.wavelength = self.wavelength
            self.non_linear_ls.build_up()
            t2 = time.time()
            if self.has_gradient_converged_to_zero(): break
            self.do_damping(self.damping_value)
            self.non_linear_ls.solve()
            if self.had_too_small_a_step() or self.analyse_shifts(self.max_shift_over_esd):
                break
            self.non_linear_ls.step_forward()
            logging.debug("-- " + "{:10.5f}".format(t2-t1) + " for reset+build_up")
            logging.debug("-- " + "{:10.5f}".format(time.time()-t2) + " for damping+ls_solve+step_forward")
            self.n_iterations += 1
        logging.debug("Timing for building-up: %.3fs, iterations: %.3fs" %(
                self.non_linear_ls.normal_equations_building_time, time.time()-start_t))

    def __str__(self):
        return "Gauss-Newton with damping and shift scaling"


class levenberg_marquardt_iterations(iterations_with_shift_analysis):
    tau = 1e-3

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self.mu_history.append(value)
        self._mu = value

    def do_naive(self):
        normal_eqns_solving.naive_iterations_with_damping_and_shift_limit.do(self)
        self.non_linear_ls.build_up()

    def do(self):
        self.mu_history = flex.double()
        self.non_linear_ls.build_up()
        if self.has_gradient_converged_to_zero() or self.n_max_iterations == 0:
            return
        start_t = time.time()
        self.n_iterations = 0
        nu = 2
        a = self.non_linear_ls.normal_matrix_packed_u()
        self.mu = self.tau*flex.max(a.matrix_packed_u_diagonal())
        while self.n_iterations < self.n_max_iterations:
            self.reset_shifts()
            a.matrix_packed_u_diagonal_add_in_place(self.mu)
            objective = self.non_linear_ls.objective()
            g = -self.non_linear_ls.opposite_of_gradient()
            self.non_linear_ls.solve()
            if self.had_too_small_a_step() or self.analyse_shifts(15):
                break
            h = self.non_linear_ls.step()
            expected_decrease = 0.5*h.dot(self.mu*h - g)
            self.non_linear_ls.step_forward()
            self.n_iterations += 1
            self.non_linear_ls.build_up(objective_only=True)
            objective_new = self.non_linear_ls.objective()
            actual_decrease = objective - objective_new
            rho = actual_decrease/expected_decrease
            if rho > 0:
                if self.has_gradient_converged_to_zero(): break
                self.mu *= max(1/3, 1 - (2*rho - 1)**3)
                nu = 2
                logging.debug("Updating mu to %.3e and applying shifts" %self.mu)
            else:
                if self.mu < 1e16:
                    self.non_linear_ls.step_backward()
                    self.mu *= nu
                    nu *= 2
                    logging.debug("Increasing mu to %.3e and recalculating" % self.mu)
                else:
                    print("Refinement did not improve, no shifts applied!")
                logging.debug("Mu is too big, skip increase")
            self.non_linear_ls.build_up()
        # get proper s.u.
        self.non_linear_ls.build_up()
        logging.debug("Timing for building-up: %.3fs, iterations: %.3fs" %(
                self.non_linear_ls.normal_equations_building_time, time.time()-start_t))


def build_refinement_wrapper(solver_name):
    solvers = {
        'Gauss-Newton': naive_iterations_with_damping_and_shift_limit,
        'Levenberg-Marquardt': levenberg_marquardt_iterations,
        'NSFF': normal_eqns_solving.levenberg_marquardt_iterations
    }
    if solver_name not in solvers:
        known_solvers_str = ', '.join(f'"{solver}"' for solver in solvers.keys())
        raise NotImplementedError(f"Solver {solver_name} is not implemented. Use one of: {known_solvers_str}")

    iterations_class = solvers[solver_name]
    class orphanedRefinementWrapper(iterations_class):
        def __init__(self, normal_eqs, *args, **kwds):
            normal_eqs.iterations_object = self
            try:
                self.wavelength = normal_eqs.refinement.xray_structure.wavelength
            except:
                warnings.warn('Could not get wavelength from x-ray structure. Some features might not work')
            super(iterations_class, self).__init__(normal_eqs, *args, **kwds)
        def do(self):
            if self.n_max_iterations == 0:
                self.n_iterations = 0
                self.non_linear_ls.build_up()
                self.non_linear_ls.solve()
                self.analyse_shifts()
            else: # super(iterations_class, self) somehow does not work here - calls iterations.do()
                iterations_class.do(self)
    return orphanedRefinementWrapper

GNRefinementWrapper = build_refinement_wrapper('Gauss-Newton')
LMRefinementWrapper = build_refinement_wrapper('Levenberg-Marquardt')
NSFFRefinementWrapper = build_refinement_wrapper('NSFF')
