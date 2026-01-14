import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EmotionAdaptiveEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral']
        self.num_emotions = len(self.emotions)
        
        self.actions = ['comfort', 'celebrate', 'apologize', 'encourage', 'ignore']
        self.num_actions = len(self.actions)
        
        self.observation_space = spaces.MultiBinary(self.num_emotions)
        
        self.action_space = spaces.Discrete(self.num_actions)
        
        self.correct_responses ={
                'happy' : ['celebrate'],
                'sad'   : ['comfort'],
                'angry' : ['apologize'],
                'fear'  : ['encourage'],
            'surprise'  : ['celebrate'],
             'neutral'  : ['ignore']
        }
        
        
        self.current_emotion = 0
        self.max_steps = 100
        self.current_step = 0
        
    def step(self, action):
            
        self.current_step += 1
            
        action_name = self.actions[action]
        emotion_name = self.emotions[self.current_emotion]
            
        reward = 10 if action_name in self.correct_responses[emotion_name] else -5
                
        terminated = False
        truncated = self.current_step >= self.max_steps
            
        self.current_emotion = np.random.choice(self.num_emotions)
        obs = np.zeros(self.num_emotions, dtype= int)
        obs[self.current_emotion] = 1
            
        return obs, reward, terminated, truncated, {'emotion' : emotion_name , 'action' : action_name}
        
    def reset(self, *, seed = None, options = None):
            
        self.current_step = 0
        self.current_emotion = np.random.choice(self.num_emotions)
            
        obs = np.zeros(self.num_emotions, dtype = int)
        obs[self.current_emotion] = 1
            
        return obs , {}           
             
