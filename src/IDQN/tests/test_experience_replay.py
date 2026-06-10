from src.IDQN.experience_replay import ReplayMemory
import random

if __name__ == '__main__':
    test = ReplayMemory(maxlen=3, seed=42)
    test.append(transition=1)
    test.append(transition=2)
    test.append(transition=3)
    print(test.memory)

    print(test.sample(3))

    random.seed(42)
    print(random.sample([1, 2, 3], k=3))

    print(len(test))

    """
    Output:
    deque([1, 2, 3], maxlen=3)
    [3, 1, 2]
    [3, 1, 2]
    3
    """