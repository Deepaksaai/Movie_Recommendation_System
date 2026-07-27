"""
train_ppo.py
============

Train PPO on the Movie Recommendation Environment.
"""

import os
import random
import numpy as np
import torch

from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)

from environment import MovieRecommendationEnv


# ==========================================================
# CONFIG
# ==========================================================

SEED = 42

TOTAL_TIMESTEPS = 5_000_000

LEARNING_RATE = 3e-4

N_STEPS = 2048

BATCH_SIZE = 256

N_EPOCHS = 10

GAMMA = 0.99

GAE_LAMBDA = 0.95

CLIP_RANGE = 0.2

ENT_COEF = 0.03     # encourage exploration

VF_COEF = 0.5

MAX_GRAD_NORM = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

LOG_DIR = "logs/ppo"

CHECKPOINT_DIR = "checkpoints/ppo"

BEST_MODEL_DIR = "best_model"

FINAL_MODEL_DIR = "trained_models"

for directory in [

    LOG_DIR,

    CHECKPOINT_DIR,

    BEST_MODEL_DIR,

    FINAL_MODEL_DIR,

]:

    os.makedirs(directory, exist_ok=True)


# ==========================================================
# RANDOM SEEDS
# ==========================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# ENVIRONMENT
# ==========================================================

def make_env():

    env = MovieRecommendationEnv(

        device=DEVICE

    )
    print("Observation Shape :", env.observation_space.shape)
    print("Action Space      :", env.action_space.n)

    env = Monitor(env)

    return env


train_env = DummyVecEnv(

    [

        make_env

    ]

)

eval_env = DummyVecEnv(

    [

        make_env

    ]

)


# ==========================================================
# CALLBACKS
# ==========================================================

checkpoint_callback = CheckpointCallback(

    save_freq = 25000,

    save_path=CHECKPOINT_DIR,

    name_prefix="ppo_movie"

)

eval_callback = EvalCallback(

    eval_env,

    best_model_save_path=BEST_MODEL_DIR,

    log_path=LOG_DIR,

    eval_freq = 25000,

    deterministic=True,

    render=False,

)

callbacks = CallbackList(

    [

        checkpoint_callback,

        eval_callback

    ]

)


# ==========================================================
# POLICY NETWORK
# ==========================================================

policy_kwargs = dict(

    activation_fn=torch.nn.ReLU,

    net_arch=dict(

        pi=[1024,512,256],

        vf=[1024,512,256],

    ),

)


# ==========================================================
# BUILD PPO
# ==========================================================

print("=" * 60)

print("Building PPO Agent")

print("=" * 60)

print(f"Device : {DEVICE}")

print()

model = PPO(

    policy="MlpPolicy",

    env=train_env,

    learning_rate=LEARNING_RATE,

    n_steps=N_STEPS,

    batch_size=BATCH_SIZE,

    n_epochs=N_EPOCHS,

    gamma=GAMMA,

    gae_lambda=GAE_LAMBDA,

    clip_range=CLIP_RANGE,

    ent_coef=ENT_COEF,

    vf_coef=VF_COEF,

    max_grad_norm=MAX_GRAD_NORM,

    policy_kwargs=policy_kwargs,

    tensorboard_log=LOG_DIR,

    verbose=1,

    seed=SEED,

    device=DEVICE,

)

# ==========================================================
# TRAINING
# ==========================================================

def train():

    print()

    print("=" * 60)

    print("Starting PPO Training")

    print("=" * 60)

    print(f"Total Timesteps : {TOTAL_TIMESTEPS:,}")

    print()

    model.learn(

        total_timesteps=TOTAL_TIMESTEPS,

        callback=callbacks,

        progress_bar=True,

    )

    print()

    print("=" * 60)

    print("Training Complete")

    print("=" * 60)

    print()

    final_model_path = os.path.join(

        FINAL_MODEL_DIR,

        "ppo_final"

    )

    model.save(final_model_path)

    print(f"Final model saved to : {final_model_path}.zip")

    print(f"Best model directory : {BEST_MODEL_DIR}")

    print(f"TensorBoard logs     : {LOG_DIR}")

    print()


# ==========================================================
# QUICK EVALUATION
# ==========================================================

def evaluate():

    print("=" * 60)

    print("Running One Evaluation Episode")

    print("=" * 60)

    print()

    env = make_env()

    obs, info = env.reset()

    total_reward = 0.0

    terminated = False

    truncated = False

    step = 0

    while not (terminated or truncated):

        action, _ = model.predict(

            obs,

            deterministic=True,

        )

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(

            f"Step {step+1:02d}"

            f" | Action {int(action):3d}"

            f" | Reward {reward:.4f}"

        )

        step += 1

    print()

    print("=" * 60)

    print("Evaluation Finished")

    print("=" * 60)

    print(f"Episode Reward : {total_reward:.4f}")

    print(f"Episode Length : {step}")

    print()


# ==========================================================
# MAIN
# ==========================================================

def main():

    train()

    evaluate()

    print("=" * 60)

    print("Done!")

    print("=" * 60)

    print()

    print("Launch TensorBoard with:")

    print()

    print("tensorboard --logdir logs/ppo")

    print()


if __name__ == "__main__":

    main()