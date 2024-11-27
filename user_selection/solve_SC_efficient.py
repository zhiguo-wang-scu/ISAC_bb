"""
Wrapper for solve_relaxation module. This module implements saving results for all the problems ever solved so that redundant computations can be avoided
"""
from user_selection.solve_relaxation_mer import solve_relaxed
import numpy as np


class EfficientRelaxation:
    def __init__(self, H=None, powe_db=1.0, sigma=1.0, rho=1.0):

        self.H = H.copy()

        self.powe_db = powe_db
        self.sigma = sigma
        self.rho = rho
        self.data = {}
        self.data['node'] = []
        self.data['solution'] = []
        self.num_problems = 0
        self.num_unique_problems = 0

    def _save_solutions(self, W=None,
                        R_X=None,
                        obj=None,
                        optimal=None):
        """
        Stores the solutions in RAM as a dictionary

        Does not save duplicate solutions. For example if the node is already present in the data, it does not store.
        """

        self.data['node'].append((W.copy(), R_X.copy()))
        self.data['solution'].append((W.copy(), R_X.copy(), obj, optimal))

    def print_nodes(self):
        for item in self.data['node']:
            print(item[0] * (1 - item[1]))



    def solve_efficient(self,gamma_l,gamma_u):
        '''
        Wrapper for solving the relaxed problems for BF and RBF
        First checks whether an equivalent node problem has already been solved.
        If so, it returns the stored solution, otherwise, it computes the new solution.
        '''


        self.num_problems += 1




        optima_Gamma, optimal_W, optimal_R_X, optimal_s, optimal_objective, feas\
            = solve_relaxed(H=self.H, gamma_l=gamma_l, gamma_u=gamma_u, powe_db=self.powe_db, sigma=self.sigma, rho=self.rho)
        # self._save_solutions(W=optimal_W,
        #                 R_X=optimal_R_X,
        #                 obj=optimal_objective,
        #                 optimal=feas)
        return optima_Gamma, optimal_W, optimal_R_X, optimal_s, optimal_objective, feas

    def get_total_problems(self):
        return self.num_problems
