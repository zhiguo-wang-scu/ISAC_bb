import numpy as np
import time
from collections import namedtuple
from dataclasses import dataclass
from user_selection.observation_SC import Observation
from user_selection.bb_SC_unified import BBenv as Environment, solve_bb
from models.helper import SolverException
from user_selection.solve_relaxation_mer import solve_relaxed


# from antenna_selection.paper_experiments.greedy import greedy

parameter_fields = (
'SC', 'train_size', 'test_size', 'sigma_sq', 'weight', 'power',  'num_trials', 'timeout')
Parameters = namedtuple('Parameters', parameter_fields)
Parameters.__new__.__defaults__ = (None,) * len(Parameters._fields)

TrainParameters = namedtuple('TrainParameters',
                             ['SC', 'train_size', 'sigma_sq', 'weight', 'power'])

INVALID_TOKENS = (np.inf, None, np.nan)





@dataclass
class OracleArg:
    instance: np.array = None
    SC: bool = None
    sigma_sq: float = 1.0
    weight: float = 1.0
    power: float = 10








@dataclass
class MLArgTest:
    instance: np.array = None
    SC: bool = None
    sigma_sq: float = 1.0
    weight: float = 1.0
    power: float = 10
    policy_filepath: str = None



@dataclass
class MLArgTrain:
    instance: np.array = None
    gamma_optimal: np.array = None
    optimal_objective: float = None
    SC: bool = None
    file_count: int = None
    sigma_sq: float = 1.0
    weight: float = 1.0
    power: float = 10
    policy_filepath: str = None




def solve_bb_pool(arguments):
    try:
        _,_,Gamma_solution, objective, output_time, output_problems = solve_bb(instance=arguments.instance,
                powe_db=arguments.power,
                sigma=arguments.sigma_sq,
                rho=arguments.weight,
                max_iter=arguments.max_iter)
        return {'solution': Gamma_solution, 'objective': objective, 'time': output_time,
                'num_problems': output_problems}

    except SolverException as e:
        print('Solver Exception: ', e)
        return {'solution': None, 'objective': np.inf, 'time': 0, 'num_problems': 0}





def solve_ml_pool(arguments: MLArgTest):
    env = Environment(observation_function=Observation, epsilon=0.02)
    env.set_node_select_policy(node_select_policy_path=arguments.policy_filepath)

    env.reset(arguments.instance,
              powe_db=arguments.power, sigma=arguments.sigma_sq, rho=arguments.weight)


    start_time = time.time()
    timestep = 0

    while timestep < 1000 and len(env.nodes) > 0:
        # if (
        #         arguments.timeout is not None and time.time() - start_time > arguments.timeout) or env.bm_solver.get_total_problems() > arguments.max_problems:
        #     break
        # print('model tester timestep {},  U: {}, L: {}'.format(timestep, env.global_U, env.global_L))
        node_id = env.select_node()

        env.push_children(node_id, arguments.sigma_sq, arguments.weight, parallel=False)
        timestep = timestep + 1

        timestep = timestep + 1

    result = {'timesteps': timestep,
              'objective': env.global_U,
              'time_taken': time.time() - start_time,
              'global_L': env.global_L,
              'num_problems': env.bm_solver.get_total_problems()}
    return result

import cvxpy as cp
import numpy.linalg as LA
def SDR_multi(H, pow_db, rho, sigma_c):
    N_t, K = H.shape
    Gamma = cp.Variable(K)
    Z = cp.Variable(N_t)
    b_k = [LA.norm(H[:, k]) ** 2 for k in range(K)]
    lambda_new = cp.Variable(N_t)
    P_T = np.power(10, pow_db / 10)
    obj = -cp.sum(cp.log(1 + Gamma)) + rho * cp.sum(Z)
    constraints = [
        cp.sum(lambda_new) <= P_T,
        ]
    for i in range(N_t):
        constraints += [
            cp.bmat([[Z[i], 1], [1, lambda_new[i]]]) >> 0,
        ]
    for i in range(K):
        constraints += [
            lambda_new[i] * b_k[i] - Gamma[i] * sigma_c >= 0
        ]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False)
        # mosek_log = prob.solver_stats.mosek
        # print("MOSEK Log:", mosek_log)
    except Exception as e:
        print(e)
    optimal_objective = prob.value

    if prob.status == cp.OPTIMAL:

        feas = 1
    else:

        feas = -1
    return optimal_objective, feas

if __name__=='__main__':
    #np.random.seed(1)
    N_t, K = 5, 3

    H_random = (np.random.randn(N_t, K) + 1j * np.random.randn(N_t, K)) / np.sqrt(2)
    print("norm channel_random:", np.linalg.norm(H_random, axis=0))

