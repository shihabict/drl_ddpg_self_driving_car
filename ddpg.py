import argparse
import time
import numpy as np
import tensorflow as tf

from model.actor import ActorNetwork as ActorNetwork
from model.critic import (CriticNetwork

                          as CriticNetwork)
from gym_torcs import TorcsEnv
from util.noise import OrnsteinUhlenbeckActionNoise
from util.replay_buffer import ReplayBuffer

np.random.seed(1337)

# Enable GPU memory gr

# owth (TF2)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def play(train_indicator):
    buffer_size = 100000
    batch_size = 32
    gamma = 0.99
    tau = 0.001
    lra = 0.0001
    lrc = 0.001
    ou_sigma = 0.3

    action_dim = 1
    state_dim = 21

    episodes_num = 2000
    max_steps = 100000
    step = 0

    train_stat_file = "data/train_stat.txt"
    actor_weights_file = "data/actor.h5"
    critic_weights_file = "data/critic.h5"

    actor = ActorNetwork(state_size=state_dim, action_size=action_dim, hidden_units=(300, 600), tau=tau, lr=lra)
    critic = CriticNetwork(state_size=state_dim, action_size=action_dim, hidden_units=(300, 600), tau=tau, lr=lrc)
    buffer = ReplayBuffer(buffer_size)

    ou = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=ou_sigma * np.ones(action_dim))

    env = TorcsEnv(vision=False, throttle=False, gear_change=False)

    try:
        actor.model.load_weights(actor_weights_file)
        critic.model.load_weights(critic_weights_file)
        actor.target_model.load_weights(actor_weights_file)
        critic.target_model.load_weights(critic_weights_file)
        print("Weights loaded successfully")
    except:
        print("Cannot load weights")

    for i in range(episodes_num):
        print(f"Episode : {i} Replay buffer {len(buffer)}")

        ob = env.reset(relaunch=(i % 3 == 0))
        state = np.hstack((ob.angle, ob.track, ob.trackPos))

        total_reward = 0.
        for j in range(max_steps):
            action_predicted = actor.model(state.reshape(1, -1)) + ou()
            observation, reward, done, info = env.step(action_predicted.numpy()[0])
            state1 = np.hstack((observation.angle, observation.track, observation.trackPos))

            buffer.add((state, action_predicted.numpy()[0], reward, state1, done))

            if len(buffer) >= batch_size:
                batch = buffer.get_batch(batch_size)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])

                target_q = critic.target_model([new_states, actor.target_model(new_states)])
                y_t = rewards + gamma * np.squeeze(target_q.numpy()) * (~dones)

                if train_indicator:
                    critic.model.train_on_batch([states, actions], y_t.reshape(-1, 1))
                    with tf.GradientTape() as tape:
                        a_out = actor.model(states)
                    a_grads = critic.get_action_gradients(states, a_out)
                    actor.train(states, a_grads)
                    actor.update_target_model()
                    critic.update_target_model()

            total_reward += reward
            state = state1
            print(f"Episode {i} - Step {step} - Action {action_predicted.numpy()[0][0]:.4f} - Reward {reward:.4f}")
            step += 1
            if done:
                break

        if i % 3 == 0 and train_indicator:
            print("Saving weights...")
            actor.model.save_weights(actor_weights_file)
            critic.model.save_weights(critic_weights_file)

        tm = time.strftime("%Y-%m-%d %H:%M:%S")
        episode_stat = f"{i}-th Episode. {step} total steps. Total reward: {total_reward:.2f}. Time {tm}"
        print(episode_stat)
        with open(train_stat_file, "a") as outfile:
            outfile.write(episode_stat + "\n")

    env.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", type=int, help="train indicator", default=0)
    args = parser.parse_args()
    play(args.train)