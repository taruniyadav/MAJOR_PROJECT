from stable_baselines3 import PPO
from emotion_rl_env import EmotionAdaptiveEnv

def train_model():
    print("Training the Rl agent for Emotion Responses")
    env = EmotionAdaptiveEnv()
    model =PPO("MlpPolicy", env, verbose=1, learning_rate= 0.001)
    
    model.learn(total_timesteps = 2000)
    
    model.save("emotion_agent_model.zip")
    print("Model is saved as emotion_agent_model")
    print("Training has been completed and now you can use it for real-time")
    
if __name__ == "__main__":
    train_model()
    