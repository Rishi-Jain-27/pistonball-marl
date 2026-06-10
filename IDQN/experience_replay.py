
class ReplayMemory:
    def __init__(self, max_len, seed=None):
        """
        init memory as a deque with maxlen being maxlen

        set seed if seed is not none
        """
        pass

    def append(self, transition):
        """
        append to the memory a transiton
        """
        pass

    def sample(self, sample_size):
        """
        return a random.sample of memory with sample_size
        """
        pass

    def __len__(self):
        """
        reutnr the len of the memory — override key bc python cant natively do this
        """
        pass
