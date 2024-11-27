import torch
import torch.nn as nn
import numpy as np
import time
import sys
sys.path.append('/home/wangzhiguo/ISAC_code_final/')

from user_selection.observation_SC import Observation,LinearObservation, prob_dep_features_from_obs
from user_selection.solve_SC_efficient import EfficientRelaxation
from models.gnn_policy_SC import GNNNodeSelectionPolicy
from models.mlp_policy import MLPNodeSelectionPolicy
from models.gnn_dataset import get_graph_from_obs

import numpy.linalg as LA
class Node(object):
    def __init__(self, gamma_l=None, gamma_u=None, R_X=None, Gamma_k=None, W_k=None, Gamma_feas=None, U=False, L=False, depth=0, parent_node=None, node_index = 0):
        '''
        @params:
            R_X: hermitian matrix
            Gamma_k: SINR
            W_k: beamforming w_k*w_k^H = W_k, which is optimal for the relaxed problem
            W_feas: feasible solution for the original optimization
            U: current global upper bound
            L: current global lower bound
            depth: depth of the node from the root of the BB tree
            node_index: unique index assigned to the node in the BB tree
            parent_node: reference to the parent Node objet
            node_index: unique index to identify the node (and count them)
        TODO: This could have been a named tuple.
        '''
        self.R_X = R_X.copy()
        self.Gamma_k = Gamma_k.copy()
        self.W_k = W_k.copy()
        self.Gamma_feas = Gamma_feas.copy()
        self.U = U
        self.L = L
        self.depth = depth
        self.parent_node = parent_node
        self.node_index = node_index
        self.gamma_l = gamma_l.copy()
        self.gamma_u = gamma_u.copy()

    def copy(self):

        new_node = Node(gamma_l=self.gamma_l,
                        gamma_u=self.gamma_u,
                        R_X=self.R_X,
                        Gamma_k=self.Gamma_k,
                        W_k=self.W_k,
                        Gamma_feas=self.Gamma_feas,
                        U=self.U,
                        L=self.L,
                        depth=self.depth,
                        parent_node=None,
                        node_index = self.node_index)
        return new_node


class DefaultBranchingPolicy(object):
    '''
    Default Branching Policy: This policy returns the antenna index from the unselected antennas with the maximum power assigned.
    This is currently using Observation object in order to extract the current solution and the decided antenna set.
    (change this to Node, so the code is readable and and insensitive to change in Obervation class)
    TODO: Convert it into a function as it no longer requires storing data for future computation.
    '''

    def __init__(self):
        pass

    def select_variable(self, observation):
        Gamma_k = observation.variable_features[:, 2]
        Gamma_k_hat = observation.variable_features[:, 3]
        Gamma_max = Gamma_k - Gamma_k_hat
        Gamma_max = Gamma_max/(1+Gamma_k_hat)



        return np.argmax(Gamma_max)
class BBenv(object):
    def __init__(self, observation_function=Observation, node_select_policy_path='default', epsilon=0.001):
        '''
        Initializes a B&B environment.
        For solving several B&B problem instances, one needs to call the reset function with the problem instance parameters
        @params:
            epsilon: The maximum gap between the global upper bound and global lower bound for the termination of the B&B algorithm.
        '''
        self._is_reset = None
        self.epsilon = epsilon  # stopping criterion
        self.H = None

        self.nodes = []  # list of problems (nodes)
        self.num_nodes = 0
        self.num_active_nodes = 0
        self.all_nodes = []  # list of all nodes to serve as training data for node selection policy
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []  # list of lower bounds on the problem
        self.U_list = []  # list of upper bounds on the problem

        self.global_L = np.nan  # global lower bound
        self.global_U = np.nan  # global upper bound

        self.action_set_indices = None
        self.active_node = None  # current active node

        self.node_select_model = None
        self.node_select_policy = self.default_node_select
        if node_select_policy_path == 'default':
            self.node_select_policy = self.default_node_select
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = self.oracle_node_select
        else:
            self.node_select_model = GNNNodeSelectionPolicy()
            #self.node_select_model = MLPNodeSelectionPolicy()
            self.node_select_model.load_state_dict(torch.load(node_select_policy_path))
            self.node_select_policy = self.learnt_node_select

        self.observation_function = observation_function
        #self.node_select_policy = self.default_node_select

    def reset(self,
              instance,
              oracle_opt=None,
              powe_db = 1.0, sigma = 1.0, rho = 1.0):
        '''
        Solve new problem instance with given max_ant, min_sinr, sigma_sq, and robust_margin
        '''
        # clear all variables

        self.nodes = []  # list of problems (nodes)
        self.all_nodes = []
        self.optimal_nodes = []
        self.node_index_count = 0

        self.L_list = []  # list of lower bounds on the problem
        self.U_list = []  # list of upper bounds on the problem
        self.global_L = np.nan  # global lower bound
        self.global_U = np.nan  # global upper bound
        self.action_set_indices = None
        self.active_node = None

        self.num_nodes = 1

        self.H = instance
        self.N, self.K = self.H.shape
        self.bm_solver = EfficientRelaxation(H=self.H,
                                              powe_db=powe_db, sigma=sigma, rho=rho)
        gamma_l = np.zeros(self.K)
        P_T = np.power(10, powe_db / 10)
        gamma_u = [P_T * LA.norm(self.H[:, k]) ** 2 / sigma for k in range(self.K)]
        [Gamma, W, R_X, optimal_s, lower_bound, feas] = self.bm_solver.solve_efficient(gamma_l=gamma_l, gamma_u=gamma_u)
        self.feas = feas
        self.W = W
        self.R_X =R_X


        # number of transmitters and users

        self.Q = [np.outer(self.H[:, k], np.conj(self.H[:, k])) for k in range(self.K)]
        self._is_reset = True
        self.action_set_indices = np.arange(1, self.K)

        self.global_L = lower_bound
        self.Gamma_hat, self.global_U = self.get_feasible_Gamma(R_X=R_X, W_k=W, Gamma_k=Gamma, gamma_l=gamma_l, sigma=sigma, rho=rho)


        self.active_node = Node(gamma_l=gamma_l,gamma_u=gamma_u,
                    R_X=R_X,
                    Gamma_k=Gamma,
                    W_k=W,
                    Gamma_feas=self.Gamma_hat,
                    U=self.global_U,
                    L=lower_bound,
                    depth=1, node_index=self.node_index_count)
        self.current_opt_node = self.active_node

        self.active_node_index = 0
        self.nodes.append(self.active_node)
        self.L_list.append(lower_bound)
        self.U_list.append(self.global_U)
        self.all_nodes.append(self.active_node)

        if oracle_opt is not None:
            self.oracle_opt = oracle_opt
        else:
            self.oracle_opt = (gamma_l+gamma_u)/2

    def get_feasible_Gamma(self, R_X=None, W_k=None, Gamma_k=None, gamma_l=None, sigma=None, rho=None):
        '''
        Obtain a feasible solution for MER problem
        '''
        Gamma_hat = np.zeros(self.K)
        SINR = np.zeros(self.K)
        for k in range(self.K):
            aa = np.real(np.trace(self.Q[k]@(R_X-W_k[k])))
            Gamma_hat[k] = Gamma_k[k]*sigma + gamma_l[k]*aa
            Gamma_hat[k] = Gamma_hat[k]/(sigma+aa)
            SINR[k] = np.real(np.trace(self.Q[k] @ W_k[k]))
            SINR[k] = SINR[k] / np.real(np.trace(self.Q[k] @ (R_X - W_k[k])) + sigma)

        obj = -np.sum(np.log(1 + Gamma_hat)) + rho * np.real(np.trace(np.linalg.inv(R_X)))

        return Gamma_hat, obj
    def select_node(self):
        '''
        Default node selection method
        TODO: the fathom method has been moved from here. So the loop is not needed
        '''

        node_id = self.rank_nodes()
        self.active_node = self.nodes[node_id]
        return node_id, self.observation_function().extract(self), self.is_optimal(self.active_node)


    def push_children(self, node_id, vars_ind,sigma,rho,parallel=False):
        '''
        Creates two children and appends it to the node list. Also executes fathom condition.
        @params:
            na) to branch on
            node_id: selected node to branch on
            parallel: whether to run the node computations in parallel
        '''
        self.delete_node(node_id)
       # vars_ind = self.select_variable(self.active_node.Gamma_k.copy(),self.active_node.Gamma_feas.copy())
        if vars_ind == None:
            return

        gamma_l_left = self.active_node.gamma_l.copy()
        gamm_l_right = self.active_node.gamma_l.copy()
        gamma_u_left = self.active_node.gamma_u.copy()
        gamm_u_right = self.active_node.gamma_u.copy()
        mid = (gamma_l_left[vars_ind] + gamma_u_left[vars_ind])/2
        gamm_l_right[vars_ind] = mid
        gamma_u_left[vars_ind] = mid


        children_sets = []
        children_sets.append([gamma_l_left.copy(), gamma_u_left.copy()])
        children_sets.append([gamm_l_right.copy(), gamm_u_right.copy()])




        children_stats = []
        t1 = time.time()
        for subset in children_sets:
            children_stats.append(self.create_children(subset,sigma,rho))


        for stat in children_stats:
            U, L, R_X, W, new_node, Gamma_hat = stat
            if new_node is not None:
                self.L_list.append(L)
                self.U_list.append(U)
                self.nodes.append(new_node)
                self.all_nodes.append(new_node)

        min_L_child = min([children_stats[i][1] for i in range(len(children_stats))])
        self.global_L = min(min(self.L_list), min_L_child)
        min_U_index = np.argmin([children_stats[i][0] for i in range(len(children_stats))])
        if self.global_U > children_stats[min_U_index][0]:
            # print('node depth at global U update {}'.format(self.active_node.depth + 1))
            self.global_U = children_stats[min_U_index][0]
            self.R_X = children_stats[min_U_index][2].copy()
            self.W = children_stats[min_U_index][3].copy()
            self.Gamma_hat = children_stats[min_U_index][5].copy()
        if len(self.nodes) == 0:
            return


        # Update the global upper and lower bound
        # update the incumbent solutions




    def create_children(self, constraint_set,sigma,rho):
        '''
        Create the Node with the constraint set
        Compute the local lower and upper bounds
        return the computed bounds to the calling function to update
        '''
        gamma_l, gamma_u = constraint_set



        [Gamma, W, R_X, optimal_s, lower_bound, feas]  = self.bm_solver.solve_efficient(gamma_l=gamma_l,gamma_u=gamma_u)

        if feas>0:
            Gamma_hat, U = self.get_feasible_Gamma(R_X=R_X, W_k=W, Gamma_k=Gamma, gamma_l=gamma_l, sigma=sigma, rho=rho)



            #if lower_bound <= self.global_U:
            # create and append node
            self.node_index_count += 1
            new_node = Node(gamma_l=gamma_l, gamma_u=gamma_u, R_X=R_X, Gamma_k=Gamma, W_k=W, Gamma_feas=Gamma_hat, U=U, L=lower_bound, depth=self.active_node.depth + 1, node_index =self.node_index_count
                            )
            return U, lower_bound,  R_X, W, new_node, Gamma_hat
           # else:
             #   return np.inf, np.inf, np.zeros([self.N, self.N]), np.zeros([self.N, self.N]), None


        else:

            return np.inf, np.inf, np.zeros([self.N,self.N]), np.zeros([self.N,self.N]), None, None

    def rank_nodes(self):
        return np.argmin(self.L_list)
    def fathom_nodes(self):
        del_ind = np.argwhere(np.array(self.L_list) > self.global_U + self.epsilon)
        if len(del_ind)>0:
            del_ind = sorted(list(del_ind.squeeze(axis=1)))
            for i in reversed(del_ind):
                # print('fathomed nodes')
                self.delete_node(i)

    def fathom(self, node_id):
        if self.nodes[node_id].L > self.global_U:
            self.delete_node(node_id)
            return True
        return False
    def is_optimal(self, node, oracle_opt=None):
        if oracle_opt is None:
            oracle = self.oracle_opt
        else:
            oracle = oracle_opt

        for i in range(len(oracle)):
            if oracle[i]>node.gamma_u[i] or oracle[i]<node.gamma_l[i]:
                return False
        return True
        # if np.linalg.norm(oracle) > np.linalg.norm(node.gamma_u) or np.linalg.norm(oracle) < np.linalg.norm(node.gamma_l):
        #     return False
        # else:
        #     return True
        # if np.linalg.norm(oracle-node.Gamma_feas) < 0.01:
        #     return True
        # else:
        #     return False

    def delete_node(self, node_id):
        del self.nodes[node_id]
        del self.L_list[node_id]
        del self.U_list[node_id]

    def default_node_select(self):
        '''
        Use the node with the lowest lower bound
        '''
        return np.argmin(self.L_list)

    def is_terminal(self):
        if self.global_U - self.global_L< self.epsilon:
            return True
        else:
            return False
    def set_node_select_policy(self, node_select_policy_path='default'):
        '''
        what policy to use for node selection
        @params:
            node_select_policy_path: one of ('default', 'oracle', gnn_node_policy_parameters)
                                        'default' -> use the lowest lower bound first policy
                                        'oracle' -> select the optimal node (optimal solution should be provided in the reset function)
                                        gnn_node_policy_parameters -> If neither of the above two arguments, this method assumes
                                            that gnn classifier parameters have been provided
        '''
        if node_select_policy_path=='default':
            self.node_select_policy = 'default'
        elif node_select_policy_path == 'oracle':
            self.node_select_policy = 'oracle'
        else:
            #self.node_select_model = MLPNodeSelectionPolicy(in_features=10*3+10*4*6+4*13)
            self.node_select_model = GNNNodeSelectionPolicy()
            model_state_dict = torch.load(node_select_policy_path)
            self.node_select_model.load_state_dict(model_state_dict)
            self.node_select_policy = 'ml_model'
    def prune(self, observation):
        if isinstance(observation, Observation):
            observation = get_graph_from_obs(observation, self.action_set_indices)
        #if isinstance(observation, LinearObservation):
        #    observation = torch.from_numpy(prob_dep_features_from_obs(observation)).reshape(1, -1).to(torch.float)
        if self.node_select_policy == 'oracle':
            return not self.is_optimal(self.active_node)
        elif self.node_select_policy == 'default':
            return False
        else:
            # out = self.node_select_model(observation.antenna_features, observation.edge_index, observation.edge_attr, observation.variable_features)
            # out = out.sum()
            # out = self.sigmoid(out)


            with torch.no_grad():
                out = self.node_select_model(observation, 1)

            if out < 0.5:
                print('prune')
                return True
            else:
                # print('select')
                return False
def solve_bb(instance,
                policy_type='default',
                powe_db=1.0,
                sigma=1.0,
                rho=1.0,
                max_iter=500,
                epsilon=0.001):
    t1 = time.time()

    if policy_type == 'default':
        env = BBenv(observation_function=Observation, epsilon=epsilon)
    elif policy_type == 'gnn':
        env = BBenv(observation_function=Observation, epsilon=epsilon)
    elif policy_type == 'oracle':
        env = BBenv(observation_function=Observation, epsilon=epsilon)
        pass

    branching_policy = DefaultBranchingPolicy()


    t1 = time.time()

    env.reset(instance,
              powe_db=powe_db, sigma=sigma, rho=rho)
    timestep = 0
    done = False
    lb_list = []
    ub_list = []
    obj_list = []
    print('\ntimestep', timestep, env.global_U, env.global_L)

    while timestep < max_iter and len(env.nodes) > 0 and not done:

        env.fathom_nodes()
        if len(env.nodes) == 0:
            break

        node_id, node_feats, label = env.select_node()
        branching_var = branching_policy.select_variable(node_feats)
        done = env.push_children( node_id, branching_var, sigma,rho, parallel=False)
        timestep = timestep + 1
        lb_list.append(env.global_L)
        ub_list.append(env.global_U)
        #print("Gamma_hat:", env.Gamma_hat)
        print('\ntimestep: {}, global U: {}, global L: {}'.format(timestep, env.global_U, env.global_L))
        if env.is_terminal():
            break

    return  lb_list, ub_list, env.Gamma_hat, env.global_U, time.time() - t1, env.bm_solver.get_total_problems(), env.R_X, env.W

if __name__=='__main__':
    import numpy.linalg as LA
    import time
    import matplotlib.pyplot as plt

    # import cvxpy as cp

    #
    # print(cp.installed_solvers())

    np.random.seed(1)
    N_t, K = 6, 3
    H = (np.random.randn(N_t, K) + 1j * np.random.randn(N_t, K)) / np.sqrt(2)
    #H = H / np.sum(np.linalg.norm(H, axis=0))
    powe_db = 30
    sigma = 1
    rho = 0.1



    lb_list, ub_list,bb_gamma,global_U, times1, problems, bb_R_X, bb_W  = solve_bb(H, powe_db = powe_db, sigma=sigma, rho=rho)



