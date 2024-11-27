import numpy as np
import pickle
import torch
import gzip

class Observation(object):
    def __init__(self):
        self.antenna_features  = None # np.zeros(N, 3) # three features for each antenna
        self.variable_features = None # np.zeros(M, 15)
        self.edge_index = None
        self.edge_features     = None # np.zeros(N*M, 3)

        pass

    def extract(self, model):
        # TODO: make the observation out of the model

        self.antenna_features = np.zeros((model.N, 3))
        RR_X = model.R_X
        eig_R_X, _  = np.linalg.eig(RR_X)
        eig_R_X_active, _ = np.linalg.eig(model.active_node.R_X)
        self.antenna_features[:,0] = np.real(eig_R_X)
        self.antenna_features[:,1] = np.imag(eig_R_X)
        self.antenna_features[:,2] = np.abs(eig_R_X)
        # self.antenna_features[:,3] = np.real(eig_R_X_active)
        # self.antenna_features[:,4] = np.imag(eig_R_X_active)
        # self.antenna_features[:,5] = np.abs(eig_R_X_active)

        # edge features
        self.edge_index = np.stack((np.repeat(np.arange(model.N), model.K), np.tile(np.arange(model.K), model.N)))
        self.edge_features = np.zeros((model.K*model.N, 6))
        self.edge_features[:,0] = np.real(model.H.reshape(-1))
        self.edge_features[:,1] = np.imag(model.H.reshape(-1))
        self.edge_features[:,2] = np.abs(model.H.reshape(-1))
        for k in range(model.K):
            eig_W_k, _ = np.linalg.eig(model.active_node.W_k[k])
            self.edge_features[k*model.N:(k+1)*model.N, 3] = np.real(eig_W_k)
            self.edge_features[k * model.N:(k + 1) * model.N, 4] = np.imag(eig_W_k)
            self.edge_features[k * model.N:(k + 1) * model.N, 5] = np.abs(eig_W_k)
        local_upper_bound = 2000 if model.active_node.U == np.inf else model.active_node.U

        # construct variable features

        self.variable_features = np.zeros((model.K, 13))
        self.variable_features[:, 0] = model.active_node.gamma_l  #  lower
        self.variable_features[:, 1] = model.active_node.gamma_u  # lower
        self.variable_features[:, 2] = model.active_node.Gamma_k
        self.variable_features[:, 3] = model.active_node.Gamma_feas
        global_upper_bound = 1000 if model.global_U == np.inf else model.global_U
        self.variable_features[:, 4] = global_upper_bound
        self.variable_features[:, 5] = model.global_L
        self.variable_features[:, 6] = (local_upper_bound - global_upper_bound) < model.epsilon
        # local features

        self.variable_features[:, 7] = model.active_node.depth

        for k in range(model.K):
            Q_k = np.outer(model.H[:, k], np.conj(model.H[:, k]))

            self.variable_features[k, 8] = np.real(np.trace(Q_k @ model.active_node.W_k[k]))  # tr(Q_kW_k)
            self.variable_features[k, 9] = np.real(np.trace(Q_k @ (model.active_node.R_X - model.active_node.W_k[k])))  # sum_{i\=k}tr(Q_kW_i)

        self.variable_features[:, 10] = 0 if model.active_node.L == np.inf else model.active_node.L
        self.variable_features[:, 11] = local_upper_bound
        self.variable_features[:, 12] = model.Gamma_hat
        return self


class LinearObservation(object):
    """
    Constructs a long obervation vector for linear neural network mapping
    """

    def __init__(self):
        self.observation = None
        self.candidates  = None # np.arange(M)
        self.variable_features = None
        pass

    def extract(self, model):
        # TODO: make the observation out of the model

        self.antenna_features = np.zeros((model.N, 3))
        RR_X = model.R_X
        eig_R_X, _ = np.linalg.eig(RR_X)
        eig_R_X_active, _ = np.linalg.eig(model.active_node.R_X)
        self.antenna_features[:, 0] = np.real(eig_R_X)
        self.antenna_features[:, 1] = np.imag(eig_R_X)
        self.antenna_features[:, 2] = np.abs(eig_R_X)
        # self.antenna_features[:,3] = np.real(eig_R_X_active)
        # self.antenna_features[:,4] = np.imag(eig_R_X_active)
        # self.antenna_features[:,5] = np.abs(eig_R_X_active)

        # edge features
        self.edge_index = np.stack((np.repeat(np.arange(model.N), model.K), np.tile(np.arange(model.K), model.N)))
        self.edge_features = np.zeros((model.K * model.N, 6))
        self.edge_features[:, 0] = np.real(model.H.reshape(-1))
        self.edge_features[:, 1] = np.imag(model.H.reshape(-1))
        self.edge_features[:, 2] = np.abs(model.H.reshape(-1))
        for k in range(model.K):
            eig_W_k, _ = np.linalg.eig(model.active_node.W_k[k])
            self.edge_features[k * model.N:(k + 1) * model.N, 3] = np.real(eig_W_k)
            self.edge_features[k * model.N:(k + 1) * model.N, 4] = np.imag(eig_W_k)
            self.edge_features[k * model.N:(k + 1) * model.N, 5] = np.abs(eig_W_k)
        local_upper_bound = 2000 if model.active_node.U == np.inf else model.active_node.U

        # construct variable features

        self.variable_features = np.zeros((model.K, 13))
        self.variable_features[:, 0] = model.active_node.gamma_l  # lower
        self.variable_features[:, 1] = model.active_node.gamma_u  # lower
        self.variable_features[:, 2] = model.active_node.Gamma_k
        self.variable_features[:, 3] = model.active_node.Gamma_feas
        global_upper_bound = 1000 if model.global_U == np.inf else model.global_U
        self.variable_features[:, 4] = global_upper_bound
        self.variable_features[:, 5] = model.global_L
        self.variable_features[:, 6] = (local_upper_bound - global_upper_bound) < model.epsilon
        # local features

        self.variable_features[:, 7] = model.active_node.depth

        for k in range(model.K):
            Q_k = np.outer(model.H[:, k], np.conj(model.H[:, k]))

            self.variable_features[k, 8] = np.real(np.trace(Q_k @ model.active_node.W_k[k]))  # tr(Q_kW_k)
            self.variable_features[k, 9] = np.real(
                np.trace(Q_k @ (model.active_node.R_X - model.active_node.W_k[k])))  # sum_{i\=k}tr(Q_kW_i)

        self.variable_features[:, 10] = 0 if model.active_node.L == np.inf else model.active_node.L
        self.variable_features[:, 11] = local_upper_bound
        self.variable_features[:, 12] = model.Gamma_hat
        self.dim_feature = 3 * model.N + 6 * model.N * model.K + 13 * model.K
        return self




def prob_dep_features_from_obs(observation):
    """
    Arguments:
        observation: Observation instance (for graph)
        output: Vector of observation (with all the information from the input observation)
    """
    # use the indices of observation to extract the features
    features = np.concatenate((observation.antenna_features.reshape(-1),
                                    observation.variable_features.reshape(-1),
                                    observation.edge_features.reshape(-1)))
    return features

def prob_indep_features_from_obs(observation):
    """
    Arguments:
        observation: Observation instance (for graph)
        output: Vector of observation (with only those features from the input observation object that is problem size independent)
    List of all problem size independent features in observation object in antenna selection:
        1. [variable features 0] global lower bound
        2. [variable features 1] global upper bound
        3. [variable features 2] local_upper_bound - global_upper_bound < model.epsilon
        4. [variable features 5] active node depth
        5. [variable features 6] local lower bound
        6. [variable features 7] local upper bound
    """
    features = np.zeros(6)
    features[0] = observation.variable_features[0,0]
    features[1] = observation.variable_features[0,1]
    features[2] = observation.variable_features[0,2]
    features[3] = observation.variable_features[0,5]
    features[4] = observation.variable_features[0,6]
    features[5] = observation.variable_features[0,7]

    return features


def get_dataset_svm(sample_files, prob_size_dependent=True):
    assert len(sample_files)>0, "list cannot be of size 0"

    features = []
    labels = []
    # features = torch.zeros(len(sample_files))
    # labels = torch.zeros(len(sample_files))

    for i in range(len(sample_files)):
        with gzip.open(sample_files[i], 'rb') as f:
            sample = pickle.load(f)
        sample_observation, target = sample[0], sample[1]
        labels.append(target)
        if prob_size_dependent:
            features.append(torch.tensor(prob_dep_features_from_obs(sample_observation), dtype=torch.float32))
        else:
            features.append(torch.tensor(prob_indep_features_from_obs(sample_observation), dtype=torch.float32))


    return torch.stack(features, axis=0), torch.tensor(labels)


class LinearDataset(torch.utils.data.Dataset):
    def __init__(self, sample_files, prob_size_dependent=True):
        super().__init__()
        self.sample_files = sample_files
        self.prob_size_dependent = prob_size_dependent

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx):

        with gzip.open(self.sample_files[idx], 'rb') as f:
            sample = pickle.load(f)
        sample_observation, target = sample[0], sample[1]

        if self.prob_size_dependent:
            features = prob_dep_features_from_obs(sample_observation)
        else:
            features = prob_indep_features_from_obs(sample_observation)

        return torch.tensor(features, dtype=torch.float32),  target




