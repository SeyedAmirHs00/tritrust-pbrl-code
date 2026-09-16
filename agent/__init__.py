import abc


class Agent:
    def reset(self):
        pass
    
    @abc.abstractmethod
    def train(self, training=True):
        pass
    
    @abc.abstractmethod
    def update(self, replay_buffer, logger, step):
        pass
    
    @abc.abstractmethod
    def act(self, obs, sample=False):
        pass