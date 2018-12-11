import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque
import random
import warnings
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap
from matplotlib.ticker import MaxNLocator
warnings.filterwarnings('ignore')
import gym

# env = retro.make(game='AirStriker-Genesis')
env = gym.make('SpaceInvaders-v0')

possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())


def preprocess_frame(frame):
    # greyscaling
    gray = rgb2gray(frame)
    # crop the frame
    cropped_frame = gray[8:-12, 4:-12]
    # normalize
    normalized_frame = cropped_frame / 255.0
    # resize the frame
    preprocessed_frame = transform.resize(normalized_frame, [110, 84])  # change to the internet notebook

    return preprocessed_frame



def stack_frames(stacked_frames, next_state, state, is_new_episode):
    next_frame = preprocess_frame(next_state)
    frame = preprocess_frame(state)

    frame = np.maximum(next_frame, frame)

    if is_new_episode:
        stacked_frames.append(next_frame)
        stacked_frames.append(next_frame)
        stacked_frames.append(next_frame)
        stacked_frames.append(next_frame)

        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames


state_size = [110, 84, 4]
action_size = env.action_space.n
learning_rate = 0.00025
total_episodes = 1000
max_steps = 50000
batch_size = 64
gamma = 0.99
pretrain_length = 1000
memory_size = 10000
episode_render = False
training = True
# FIXED Q TARGETS HYPERPARAMETERS 
max_tau = 10000  # Tau is the C step where we update our target network
skiped_frame_treshold = 3
step_threshold = 10000
# SAMPLING STRATEGY FLAG: PER vs random
sampling_strategy = 'random'
stack_size = 4
stacked_frames = deque([np.zeros((110, 84), dtype=int) for i in range(stack_size)], maxlen=4)


class DQNetSimple(object):
    def __init__(self, state_size, action_size, learning_rate, name='DQNetworkSimple'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            #
            self.inputs_ = tf.placeholder(tf.float32, [None, (*state_size)], name='inputs')
            #
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name='actions_')
            #
            self.target_Q = tf.placeholder(tf.float32, [None], name='target')
            #
            self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='ISWeights_')

            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, filters=32, kernel_size=[8, 8], strides=[4, 4],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv1')
            #self.conv1 = tf.layers.batch_normalization(self.conv1, axis=-1,momentum=0.95)#introduce batch normalization
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2')
            #self.conv2 = tf.layers.batch_normalization(self.conv2, axis=-1,momentum=0.95)#introduce batch normalization
            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64, kernel_size=[3, 3], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv3')
            #self.conv3 = tf.layers.batch_normalization(self.conv3, axis=-1,momentum=0.95)#introduce batch normalization
            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            #self.fc = tf.layers.batch_normalization(self.fc, axis=-1,momentum=0.95)#introduce batch normalization
            self.output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size, activation=None)

            self.final_output = tf.identity(self.output, name='final_output')

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # COMPUTE HUBER LOSS
            self.delta = self.target_Q - self.Q
            self.loss = tf.reduce_mean(self.clipped_error(self.delta), name='loss')
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def clipped_error(self, x):
        # Huber loss
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x))


def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class RandomMemory(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=True)

        return [self.buffer[i] for i in index]


def init_memory_random(stacked_frames):
    memory = RandomMemory(memory_size)
    state = env.reset()
    state, stacked_frames = stack_frames(stacked_frames, state, state, True)

    for i in range(pretrain_length):

        choice = random.randint(1, len(possible_actions)) - 1

        next_state, reward, done, _ = env.step(choice)

        next_state, stacked_frames = stack_frames(stacked_frames, next_state, state, False)

        if done:
            next_state = np.zeros(state.shape)
            # Prio MEMORY also expects error
            if reward > 0:
                memory.add((state, choice, 1, next_state, done))
            else:
                memory.add((state, choice, 0, next_state, done))
            #else:
             #   memory.add((state, choice, 0, next_state, done))
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, next_state, state, True)
        else:
            # Prio MEMORY also expects error
            #if reward != 0:
            if reward > 0:
                memory.add((state, choice, 1, next_state, done))
            else:
                memory.add((state, choice, 0, next_state, done))
            #else:
             #   memory.add((state, choice, 0, next_state, done))
            state = next_state

    return memory


# Instantiate memory
if sampling_strategy == 'PER':
    pass
else:
    memory = init_memory_random(stacked_frames)


def predict_action(sess, DQNetwork, decay_step, state, epsilon):
    exp_exp_tradeoff = np.random.rand()

    if decay_step < step_threshold:
        epsilon -= (skiped_frame_treshold + 1) * 0.00009

    if exp_exp_tradeoff < epsilon:
        choice = random.randint(1, len(possible_actions)) - 1
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)

    return choice, epsilon



"""
def save_performance_statistics(episodes, steps_per_episode, total_rewards):
  plot_steps_total_rewards(episodes, steps_per_episode, total_rewards)
  plot_average_step_value(episodes, steps_per_episode, total_rewards)

def plot_average_step_value(episodes, steps_per_episode, total_rewards):
  average = np.array(total_rewards)/ np.array(steps_per_episode)
  plt.plot(episodes, average, color='maroon', marker='o')
  plt.xlabel('Episodes', fontsize=15)
  ax = plt.gca()
  ax.set_facecolor('white')
  ax.legend(['Average reward per episode'], fontsize=15)
  plt.xticks(size = 15)
  plt.yticks(size = 15)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.savefig('episodes_average_step_value.png')
  plt.gcf().clear()

def plot_steps_total_rewards(episodes, steps_per_episode, total_rewards):
  plt.plot(episodes, steps_per_episode, color='seagreen', marker='o')
  plt.plot(episodes, total_rewards, color='maroon', marker='o')
  plt.xlabel('Episode', fontsize=15)
  ax = plt.gca()
  ax.set_facecolor('white')
  ax.legend(['Number of steps', 'Total rewards'], fontsize=15)
  plt.xticks(size = 15)
  plt.yticks(size = 15)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  #plt.show()
  plt.savefig('episodes_steps_rewards_plot.png')
  plt.gcf().clear()
"""
def train_simple(stacked_frames):
    epsilon = 1
    highest_reward = 0
    #
    tf.reset_default_graph()

    # Instantiate the DQNetwork
    DQNetwork = DQNetSimple(state_size, action_size, learning_rate, name="DQNetwork")

    # Instantiate the target network
    DQNetworkClone = DQNetSimple(state_size, action_size, learning_rate, name="TargetNetwork")

    saver = tf.train.Saver()

    episodes_stats, steps_per_episode_stats, total_rewards_per_episode_stats = [], [], []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('log' + '/original_train', sess.graph)
        tau = 0
        decay_step = 0

        skipped_frames = 0
        action = random.randint(1, len(possible_actions)) - 1 #initialization of the first actions otherwise an error of initialization occurs!

        reward_list = []

        update_target = update_target_graph()
        sess.run(update_target)

        average_reward_list = []

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, state, True)

            # used for death punishment
            flag1 = 0
            flag2 = 0

            print(epsilon)

            while step < max_steps:
                decay_step += 1
                tau += 1
                step += 1
                skipped_frames += 1


                if skipped_frames > skiped_frame_treshold:
                    action, epsilon = predict_action(sess, DQNetwork, decay_step, state, epsilon)
                    skipped_frames = 0


                next_state, reward, done, death = env.step(action)

                if episode_render:
                    env.render()

                episode_rewards.append(reward)

                if done:
                    #episodes_stats.append(episode)
                    #steps_per_episode_stats.append(step)
                    #total_rewards_per_episode_stats.append(np.sum(episode_rewards))

                    next_state = np.zeros((110, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, state, False)

                    total_reward = np.sum(episode_rewards)

                    average_episode_reward = total_reward / step

                    average_reward_list.append(average_episode_reward)

                    total_reward = np.sum(episode_rewards)
                    if average_episode_reward >= np.max(average_reward_list) and epsilon < 0.2:
                        #save_path = saver.save(sess, "./model_trained.ckpt")
                        saver.save(sess, "./model_trained.ckpt")
                        print("A better model has been found and saved!")


                    print('Episode:{}'.format(episode), 'Total reward:{}'.format(total_reward), 'Average_reward:{}'.format(average_episode_reward), 'Training Loss:{}'.format(loss))
                    reward_list.append((episode, total_reward))
                    #if reward != 0:
                    if reward > 0:
                        memory.add((state, action, 1, next_state, done))
                    else:
                        memory.add((state, action, 0, next_state, done))
                    #else:
                     #   memory.add((state, action, 0, next_state, done))
                    step = max_steps
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, state, False)
                    """
                    # death punishment
                    if death == {'ale.lives': 2} and flag2 == 0:
                        print("Punished for the first death!")
                        reward -= 1000
                        flag2 = 1

                    # death punishment
                    if death == {'ale.lives': 1} and flag1 == 0:
                        print("Punished for the second death!")
                        reward -= 1000
                        flag1 = 1
                    """
                    #if reward != 0:
                    if reward > 0:
                        memory.add((state, action, 1, next_state, done))
                    else:
                        memory.add((state, action, 0, next_state, done))
                    #else:
                     #   memory.add((state, action, 0, next_state, done))
                    state = next_state


                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])


                #print(rewards_mb)

                target_Qs_batch = []

                Qs_theta_minus = sess.run(DQNetworkClone.output, feed_dict={DQNetworkClone.inputs_: next_states_mb})
                #print(Qs_theta_minus)
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]
                    if terminal == True:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        target = rewards_mb[i] + gamma * np.max(Qs_theta_minus[i])
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])
                actions_mb_new = []

                for element in actions_mb:
                    empty_list = np.zeros(action_size)
                    empty_list[element] = 1
                    actions_mb_new.append(empty_list)

                actions_mb_new = np.array(actions_mb_new)


                loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer], feed_dict={DQNetwork.inputs_: states_mb, DQNetwork.target_Q: targets_mb, DQNetwork.actions_: actions_mb_new})


                """
                final_network_outputs, final_actions_values, final_computed_q, final_target_q, loss, _ = sess.run(
                    [DQNetwork.final_output, DQNetwork.actions_, DQNetwork.Q, DQNetwork.target_Q, DQNetwork.loss,
                     DQNetwork.optimizer],
                    feed_dict={DQNetwork.inputs_: states_mb, DQNetwork.target_Q: targets_mb,
                               DQNetwork.actions_: actions_mb_new})


                print("The are the rewards: ", rewards_mb )
                print("These are the targets: ", targets_mb)
                print('Final target Q values')
                print(final_target_q)
                print()
                """

                if tau > max_tau:
                    print("Target updated!")
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0

                #save_performance_statistics(episodes_stats, steps_per_episode_stats, total_rewards_per_episode_stats)


if training == True:
    if sampling_strategy == 'PER':
        print('Training with Prioritized Experience Replay...')

    else:
        print('Training basic DQN...')
        train_simple(stacked_frames)
