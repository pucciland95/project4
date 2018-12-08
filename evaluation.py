import tensorflow as tf
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
import random
import warnings
from time import sleep

warnings.filterwarnings('ignore')
import gym

env = gym.make('SpaceInvaders-v0')


def preprocess_frame(frame):
    # greyscaling
    gray = rgb2gray(frame)
    # crop the frame
    cropped_frame = gray[8:-12, 4:-12]
    # normalize
    normalized_frame = cropped_frame / 255.0  #####CHANGED
    # resize the frame
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])  # change to the internet notebook

    return preprocessed_frame


stack_size = 4
stacked_frames = deque([np.zeros((110, 84), dtype=int) for i in range(stack_size)], maxlen=4)


def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)

    if is_new_episode:
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


state_size = [110, 84, 4]
action_size = env.action_space.n
decay_rate = 0.00001
gamma = 0.9
stack_size = 4
epsilon = 0.01

with tf.Session() as sess:
    total_rewards = 0
    total_test_rewards = []

    saver = tf.train.import_meta_graph('model_trained.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    graph = tf.get_default_graph()
    writer = tf.summary.FileWriter(logdir="logdir", graph=graph)
    writer.flush()

    pred = graph.get_tensor_by_name('DQNetwork/final_output:0')
    print('DQNetwork/prediction from graph', pred)
    for episode in range(1):
        gym_old_state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, gym_old_state, True)
        print("*****************************")
        print("EPISODE", episode)
        print('variables')
        # for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        #     print(var)
        #     print(var.name)

        while True:
            sleep(0.05)
            random_number = np.random.rand()

            if random_number > epsilon:
                Qs = sess.run([pred], feed_dict={"DQNetwork/inputs:0": state.reshape((1, *state.shape))})
                choice = np.argmax(Qs)
            else:
                choice = random.randint(0, action_size - 1)
            next_state_gym, reward, done, _ = env.step(choice)
            #print("Gym State equality check", np.array_equal(gym_old_state, next_state_gym))

            env.render()
            total_rewards += reward
            if done:
                print("Score:", total_rewards)
                total_test_rewards.append(total_rewards)
                break
            next_state, stacked_frames = stack_frames(stacked_frames, next_state_gym, False)
            #print("Stacked State equality check", np.array_equal(state, next_state))
            state = next_state
            gym_old_state = next_state_gym