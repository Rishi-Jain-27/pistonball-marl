from src.IDQN.dqn import DQN
import torch

if __name__ == '__main__':
    state_dim = 2
    action_dim = 2
    hidden_dim = 2
    torch.manual_seed(42)
    test = DQN(state_dim=state_dim,
               action_dim=action_dim,
               hidden_dim=hidden_dim)
    test_input = torch.randn(1, state_dim) # first dim is for batching

    output = test(test_input)

    print(output)
    """
    Output:
    tensor([[0.9014, 0.3143]], grad_fn=<AddmmBackward0>)
    """