import torch
import torch_geometric

LEARNING_RATE = 0.001
NB_EPOCHS = 50
PATIENCE = 10
EARLY_STOPPING = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_SIZE = 32



class MLPNodeSelectionPolicy(torch.nn.Module):
    def __init__(self, in_features):
        super(MLPNodeSelectionPolicy, self).__init__()
        self.in_features = in_features

        # Hidden layer
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, EMB_SIZE),
            torch.nn.ReLU(),
            torch.nn.Linear(EMB_SIZE, EMB_SIZE),
            torch.nn.ReLU(),
            # torch.nn.Linear(256, EMB_SIZE),
            # torch.nn.ReLU()
        )

        # Output layer
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(EMB_SIZE, 1, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, obs, batch_size):
        hidden_output = self.hidden_layer(obs)
        logits = self.output_module(hidden_output)

        logits = logits.view(batch_size, -1).mean(dim=1)
        return logits

