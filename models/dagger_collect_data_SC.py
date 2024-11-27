from user_selection.observation_SC import Observation
from user_selection.bb_SC_unified import BBenv as Environment, DefaultBranchingPolicy, solve_bb

import numpy as np
import matplotlib
matplotlib.use('Agg')
import gzip
import pickle
from pathlib import Path
import time
import os
from torch.multiprocessing import Pool
import numpy.linalg as LA
from models.helper import SolverException
from torch.utils.data import Dataset
from user_selection.utils_SC import MLArgTrain, OracleArg, TrainParameters
import matplotlib.pyplot as plt
MAX_STEPS = 500


class OracleDataset(Dataset):
    def __init__(self, root=None):
        self.sample_files = [str(path) for path in Path(root).glob('sample_*.pkl')]
        self.save_file_index = len(self.sample_files)
        self.fetch_file_index = 0
        self.root = root

    def re_init(self):
        self.sample_files = [str(path) for path in Path(self.root).glob('sample_*.pkl')]
        self.save_file_index = len(self.sample_files)
        self.fetch_file_index = 0

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)
            # Sample expected of format H, solution, objective, timesteps, time
            return sample

    def get_batch(self, batch_size):
        sample_list = []
        if not len(self.sample_files) - self.fetch_file_index >= batch_size:
            return None
        for i in range(self.fetch_file_index, self.fetch_file_index + batch_size):
            sample_list.append(self.__getitem__(i))
        self.fetch_file_index += batch_size
        return zip(*sample_list)

    def get_batch_from_indices(self, index_list):
        sample_list = []
        for i in index_list:
            sample_list.append(self.__getitem__(i))
        return zip(*sample_list)

    def save_batch(self,
                   instances,
                   optimal_solution,
                   optimal_obj,
                   oracle_steps,
                   oracle_time,
                   lower_bound,
                   upper_bound):

        for i in range(len(instances)):
            self.put((instances[i], optimal_solution[i], optimal_obj[i], oracle_steps[i], oracle_time[i], lower_bound[i], upper_bound[i]))

    def put(self, sample):
        with gzip.open(os.path.join(self.root, 'sample_{}.pkl'.format(self.save_file_index)), 'wb') as f:
            pickle.dump(sample, f)
        self.save_file_index += 1


class DataCollect(object):
    def __init__(self,
                observation_function=Observation,
                parameters: TrainParameters = None,
                train_filepath=None,
                policy_type='gnn',
                oracle_solution_filepath=None,
                num_instances=10):

        self.observation_function = observation_function
        self.parameters = parameters


        self.policy_type = policy_type
        self.node_select_policy = None

        self.filepath = train_filepath

        if not os.path.isdir(oracle_solution_filepath):
            Path(oracle_solution_filepath).mkdir(exist_ok=True)
        self.oracle_problem_index = 0
        self.oracle_dataset = OracleDataset(root=oracle_solution_filepath)

        self.num_instances = num_instances
        self.file_count_offset = 0

    def collect_data(self, num_instances=10, policy='oracle', train=True, iter_count= 10):

        N, M = self.parameters.train_size

        # fetch the following data from saved files or create new if all are used up
        instances = None
        optimal_solution_list = []
        optimal_objective_list = []
        avg_oracle_num_problems = []
        avg_oracle_time = []
        lb_list_oracle =[]
        ub_list_oracle =[]
        lb_list_ml =[]
        ub_list_ml =[]
        # For training new data is needed in each iteration
        if train:
            samples = self.oracle_dataset.get_batch(self.num_instances)
        else:
            self.oracle_dataset.re_init()
            if self.num_instances > len(self.oracle_dataset):
                samples = None
            else:
                samples = self.oracle_dataset.get_batch_from_indices(list(range(self.num_instances)))

        if samples is not None:
            instances, optimal_solution_list, optimal_objective_list, oracle_steps, oracle_time, lb_list_oracle, ub_list_oracle = samples
            instances = np.stack(instances, axis=0)
            avg_oracle_num_problems = np.mean(oracle_steps)
            avg_oracle_time = np.mean(oracle_time)

        else:

            instances = (np.random.randn(num_instances, N, M) + 1j * np.random.randn(num_instances, N, M)) / np.sqrt(2)


            arguments_oracle = [OracleArg(instance=instances[i],
                                          SC=self.parameters.SC,
                                          sigma_sq=self.parameters.sigma_sq,
                                          weight = self.parameters.weight,
                                          power = self.parameters.power) for i in range(len(instances))]

            # arguments_oracle = list(zip(list(instances), [max_ant]*num_instances))
            print('starting first pool')
            with Pool(min(num_instances, 10)) as p:
                out_oracle = p.map(self.solve_bb_process, arguments_oracle)
                print('first pool ended')

            # Prune away the problem instances that were not feasible (could not be solved)
            for i in range(len(out_oracle) - 1, -1, -1):
                if out_oracle[i]['objective'] == np.inf:
                    del out_oracle[i]
                    instances = np.concatenate((instances[:i, ::], instances[i + 1:, ::]), axis=0)

            # the returned order is x_opt:[can be a tuple], global_U, timsteps, time
            optimal_solution_list = [out_oracle[i]['solution'] for i in range(len(out_oracle))]
            optimal_objective_list = [out_oracle[i]['objective'] for i in range(len(out_oracle))]
            avg_oracle_num_problems = np.mean(np.array([out_oracle[i]['num_problems'] for i in range(len(out_oracle))]))
            avg_oracle_time = np.mean(np.array([out_oracle[i]['time'] for i in range(len(out_oracle))]))

            self.oracle_dataset.save_batch(list(instances),
                                           [out_oracle[i]['solution'] for i in range(len(out_oracle))],
                                           [out_oracle[i]['objective'] for i in range(len(out_oracle))],
                                           [out_oracle[i]['num_problems'] for i in range(len(out_oracle))],
                                           [out_oracle[i]['time'] for i in range(len(out_oracle))],
                                           [out_oracle[i]['lb_list'] for i in range(len(out_oracle))],
                                           [out_oracle[i]['ub_list'] for i in range(len(out_oracle))])
            lb_list_oracle = [out_oracle[i]['lb_list'] for i in range(len(out_oracle))]
            ub_list_oracle = [out_oracle[i]['ub_list'] for i in range(len(out_oracle))]
        # arguments_ml = list(zip(list(instances), optimal_solution_list, optimal_objective_list, range(len(instances)),
        #                         [policy] * len(instances)))
        P_T = np.power(10, self.parameters.power / 10)
        arguments_ml = [MLArgTrain(instance=instances[i],
                                   gamma_optimal=optimal_solution_list[i],
                                   optimal_objective = optimal_objective_list[i],
                                   SC = self.parameters.SC,
                                   file_count = i,
                                   sigma_sq = self.parameters.sigma_sq,
                                   weight = self.parameters.weight,
                                   power=self.parameters.power,
                                   policy_filepath=policy,
                                   ) for i in range(len(instances))
                        ]

        print('starting second pool')
        with Pool(min(len(instances), 10)) as p:
            out_ml = p.map(self.collect_data_instance, arguments_ml)
            print('second pool ended')

        # the returned order for collect_data_instance is timesteps, ogap, time (seconds), ratio of optimal nodes to total nodes
        avg_ml_num_problems = np.mean(np.array([out_ml[i]['num_problems'] for i in range(len(out_ml))]))
        lb_list_ml= [out_ml[i]['lb_list'] for i in range(len(out_ml))]
        ub_list_ml = [out_ml[i]['ub_list'] for i in range(len(out_ml))]
        avg_ml_ogap = 0
        num_solved = 0
        for i in range(len(out_ml)):
            if out_ml[i]['ogap'] > -1:
                avg_ml_ogap += out_ml[i]['ogap']
                num_solved += 1
        avg_ml_ogap /= num_solved
        # avg_ml_ogap = np.mean(np.array([out_ml[i][1] for i in range(len(out_ml))]))
        avg_ml_time = np.mean(np.array([out_ml[i]['time'] for i in range(len(out_ml))]))

        if train:
            self.file_count_offset += len(instances)
        if train is False:
            fig = plt.figure(figsize=(4, 4))
            plt.grid(color="k", linestyle=":")
            plt.plot(lb_list_oracle[0], 'b-.', label='lower bound B&B')
            plt.plot(ub_list_oracle[0], 'g-',label='upper bound B&B')
            plt.plot(lb_list_ml[0],'r-.', label='lower bound ML B&B')
            plt.plot(ub_list_ml[0], 'c-',label='upper bound ML B&B')
            plt.xlabel('Iteration number')
            plt.ylabel('Objective value')
            plt.legend()
            plt.savefig("plot{}.png".format(iter_count), bbox_inches='tight')
            plt.close()
        # return order is time speedup, ogap, steps_speedup
        return {'time_speedup': avg_oracle_time / avg_ml_time, 'ogap': avg_ml_ogap,
                'problems_speedup': avg_oracle_num_problems / avg_ml_num_problems, 'lb_list_oracle': lb_list_oracle,
                'ub_list_oracle': ub_list_oracle, 'lb_list_ml': lb_list_ml, 'ub_list_ml': ub_list_ml}


    def collect_data_instance(self, arguments: MLArgTrain):

        print('function {} started'.format(arguments.file_count))
        # TODO: do the following with parameters not filename
        # print('optimal ', w_optimal)
        env = Environment(observation_function=self.observation_function)

        env.set_node_select_policy(node_select_policy_path=arguments.policy_filepath)

        env.reset(arguments.instance,
             oracle_opt=arguments.gamma_optimal,
              powe_db=arguments.power, sigma=arguments.sigma_sq, rho=arguments.weight)

        branching_policy = DefaultBranchingPolicy()
        t1 = time.time()
        timestep = 0
        done = False
        time_taken = 0
        sum_label = 0
        node_counter = 0
        lb_list = []
        ub_list = []
        while timestep < MAX_STEPS and len(env.nodes) > 0 and not done:
            print('timestep {}'.format(timestep))
            env.fathom_nodes()
            if len(env.nodes) == 0:
                break
            node_id, node_feats, label = env.select_node()

            if len(env.nodes) == 0:
                break
            time_taken += time.time() - t1
            sum_label += label
            self.save_file((node_feats, label), arguments.file_count, node_counter)
            node_counter += 1
            t1 = time.time()
            prune_node = env.prune(node_feats)

            if prune_node:
                env.delete_node(node_id)
                continue
            else:
                last_id = len(env.nodes)


                try:
                    branching_var = branching_policy.select_variable(node_feats)
                    done = env.push_children(node_id, branching_var, sigma=arguments.sigma_sq, rho=arguments.weight,parallel=False)
                except:
                    break
            #branching_var = branching_policy.select_variable(node_feats)
            #done = env.push_children(node_id, branching_var, sigma=arguments.sigma_sq, rho=arguments.weight,
            #                         parallel=False)
            lb_list.append(env.global_L)
            ub_list.append(env.global_U)
            timestep = timestep + 1
            if env.is_terminal():
                break

        # if node_counter < 1:
        #     print('node counter null H {}, w_opt {}'.format(env.H_complex, arguments.w_optimal))

        ml = env.global_U
        ogap = (ml - arguments.optimal_objective)/np.absolute(arguments.optimal_objective)

        print(
            'instance result: timestep {}, ogap {}, time {}, sum_label {}, optimal objective {}, ml {}'.format(timestep,
                                                                                                               ogap,
                                                                                                               time_taken,
                                                                                                               sum_label,
                                                                                                               arguments.optimal_objective,
                                                                                                               ml))

        # return order is timesteps, ogap, time (seconds), ratio of optimal nodes to total nodes
        return {'timestep': timestep, 'ogap': ogap, 'time': time_taken, 'optimal_node_ratio': sum_label / node_counter,
                'num_problems': env.bm_solver.num_problems,'lb_list': lb_list, 'ub_list': ub_list}

    def solve_bb_process(self, arguments: OracleArg):
        try:
            lb_list, ub_list, Gamma_solution, objective, output_time, output_problems, output_R_X, output_W = solve_bb(instance=arguments.instance,
                                                                                     powe_db=arguments.power,
                                                                                     sigma=arguments.sigma_sq,
                                                                                     rho=arguments.weight)
            return {'solution': Gamma_solution, 'objective': objective, 'time': output_time,
                    'num_problems': output_problems, 'lb_list': lb_list, 'ub_list': ub_list}
        except SolverException as e:
            print('Solver Exception: ', e)
            return {'solution': None, 'objective': np.inf, 'time': 0, 'num_problems': 0}

    def save_file(self, sample, file_count, node_counter):
        if self.filepath is not None:
            filename = os.path.join(self.filepath,
                                    'sample_{}_{}.pkl'.format(self.file_count_offset + file_count, node_counter))
            with gzip.open(filename, 'wb') as f:
                pickle.dump(sample, f)

    # def dummy_collect_instance(self, arguments):
    #     instance, w_optimal, optimal_objective, file_count = arguments
    #     print('started collect instance {}'.format(file_count))
    #     import time
    #     time.sleep(1)
    #     print('ended collect instance {}'.format(file_count))