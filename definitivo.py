import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage import transform
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from collections import deque
import random
import warnings

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
learning_rate = 0.00025
total_episodes = 200
max_steps = 50000
batch_size = 32
explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.00001
gamma = 0.9
pretrain_length = batch_size
memory_size = 1000000
stack_size = 4
episode_render = True
training = True
c_threshold = 4
epsilon = 0.05
# FIXED Q TARGETS HYPERPARAMETERS 
max_tau = 10000  # Tau is the C step where we update our target network

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
            # self.conv1 = tf.layers.batch_normalization(self.conv1, axis=-1,momentum=0.99)#introduce batch normalization
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2')
            # self.conv2 = tf.layers.batch_normalization(self.conv2, axis=-1,momentum=0.99)#introduce batch normalization
            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64, kernel_size=[3, 3], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv3')
            # self.conv3 = tf.layers.batch_normalization(self.conv3, axis=-1,momentum=0.99)#introduce batch normalization
            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            # self.fc = tf.layers.batch_normalization(self.fc, axis=-1,momentum=0.99)#introduce batch normalization
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


class DQNet(object):
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
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
            self.conv1 = tf.layers.batch_normalization(self.conv1, axis=-1,
                                                       momentum=0.99)  # introduce batch normalization
            self.conv1_out = tf.nn.relu(self.conv1, name="conv1_out")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, filters=64, kernel_size=[4, 4], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv2')
            self.conv2 = tf.layers.batch_normalization(self.conv2, axis=-1,
                                                       momentum=0.99)  # introduce batch normalization
            self.conv2_out = tf.nn.relu(self.conv2, name="conv2_out")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, filters=64, kernel_size=[3, 3], strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name='conv3')
            self.conv3 = tf.layers.batch_normalization(self.conv3, axis=-1,
                                                       momentum=0.99)  # introduce batch normalization
            self.conv3_out = tf.nn.relu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.layers.dense(inputs=self.flatten, units=512, activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.fc = tf.layers.batch_normalization(self.fc, axis=-1, momentum=0.95)  # introduce batch normalization
            self.output = tf.layers.dense(inputs=self.fc, kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units=self.action_size, activation=None)

            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


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


class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0

    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """

    def __init__(self, capacity):
        self.capacity = capacity  # Number of leaf nodes (final nodes) that contains experiences

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """

    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0

    """
    Update the leaf priority score and propagate the change through tree
    """

    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:  # this method is faster than the recursive loop in the reference code

            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0

        while True:  # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


class PERMemory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx = np.empty((n,), dtype=np.int32)
        b_ISWeights = np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        weights_before_normalization = []

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            weights_before_normalization.append(np.power(n * sampling_probabilities, -self.PER_b))
            # b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight
            b_idx[i] = index
            experience = [data]
            memory_b.append(experience)

        max_weight = np.max(weights_before_normalization)

        for i in range(len(weights_before_normalization)):
            b_ISWeights[i, 0] = weights_before_normalization[i] / max_weight

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


def init_memory_random(stacked_frames):
    memory = RandomMemory(memory_size)

    for i in range(pretrain_length):
        if i == 0:
            state = env.reset()

            state, stacked_frames = stack_frames(stacked_frames, state, True)

            choice = random.randint(1, len(possible_actions)) - 1
            action = possible_actions[choice]

            next_state, reward, done, _ = env.step(choice)

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            if done:
                next_state = np.zeros(state.shape)
                # Prio MEMORY also expects error
                memory.add((state, choice, reward, next_state, done))
                state = env.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True)
            else:
                # Prio MEMORY also expects error
                memory.add((state, choice, reward, next_state, done))
                state = next_state
    return memory


def init_memory_per(stacked_frames):
    memory = PERMemory(memory_size)

    for i in range(pretrain_length):
        # If it's the first step
        if i == 0:
            # First we need a state
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        # Random action
        choice = random.randint(1, len(possible_actions)) - 1
        action = possible_actions[choice]

        # Get the rewards
        next_state, reward, done, _ = env.step(choice)

        # If we're dead
        if done:
            # We finished the episode
            next_state = np.zeros(state.shape)

            # Add experience to memory
            # experience = np.hstack((state, [action, reward], next_state, done))

            experience = state, action, reward, next_state, done
            memory.store(experience)

            # Start a new episode
            # game.new_episode()

            # First we need a state
            state = env.reset()

            # Stack the frames
            state, stacked_frames = stack_frames(stacked_frames, state, True)

        else:
            # Get the next state
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            # Add experience to memory
            experience = state, action, reward, next_state, done
            memory.store(experience)

            # Our state is now the next_state
            state = next_state
        return memory


# Instantiate memory
if sampling_strategy == 'PER':
    memory = init_memory_per(stacked_frames)
else:
    memory = init_memory_random(stacked_frames)


def predict_action(sess, DQNetwork, state):
    exp_exp_tradeoff = np.random.rand()

    if exp_exp_tradeoff < epsilon:
        choice = random.randint(0, len(possible_actions) - 1)
    else:
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        choice = np.argmax(Qs)

    return choice


def train_simple(stacked_frames):
    highest_reward = 0
    #
    tf.reset_default_graph()

    # Instantiate the DQNetwork
    DQNetwork = DQNetSimple(state_size, action_size, learning_rate, name="DQNetwork")

    # Instantiate the target network
    DQNetworkClone = DQNetSimple(state_size, action_size, learning_rate, name="TargetNetwork")

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter("/tensorboard/dqn/1")

    ## Losses
    tf.summary.scalar("Loss", DQNetwork.loss)

    write_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('log' + '/original_train', sess.graph)
        tau = 0
        decay_step = 0
        reward_list = []

        update_target = update_target_graph()
        sess.run(update_target)
        average_reward_list = []

        for episode in range(total_episodes):
            step = 0
            episode_rewards = []
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                tau += 1
                step += 1
                action = predict_action(sess, DQNetwork, state)
                next_state, reward, done, _ = env.step(action)

                if episode_render:
                    env.render()

                episode_rewards.append(reward)

                if done:
                    next_state = np.zeros((110, 84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    total_reward = np.sum(episode_rewards)

                    average_episode_reward = total_reward / step

                    average_reward_list.append(average_episode_reward)

                    total_reward = np.sum(episode_rewards)
                    if average_episode_reward > np.max(average_reward_list):
                        save_path = saver.save(sess, "./model_trained.ckpt")
                        print("A better model has been found and saved!")


                    print('Episode:{}'.format(episode), 'Total reward:{}'.format(total_reward), 'Average_reward:{}'.formt(average_episode_reward),'Training Loss:{}'.format(loss))
                    reward_list.append((episode, total_reward))
                    memory.add((state, action, reward, next_state, done))
                    step = max_steps
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    memory.add((state, action, reward, next_state, done))
                    state = next_state

                batch = memory.sample(batch_size)
                states_mb = np.array([each[0] for each in batch], ndmin=3)
                actions_mb = np.array([each[1] for each in batch])
                rewards_mb = np.array([each[2] for each in batch])
                next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                dones_mb = np.array([each[4] for each in batch])

                target_Qs_batch = []

                Qs_theta_minus = sess.run(DQNetworkClone.output, feed_dict={DQNetworkClone.inputs_: next_states_mb})
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

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0

                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb, DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb_new})
                writer.add_summary(summary, episode)
                writer.flush()


def train_per(stacked_frames):
    highest_reward = 0
    #
    tf.reset_default_graph()
    # DQNetwork = DQNetwork(state_size,action_size,learning_rate)

    # Instantiate the DQNetwork
    DQNetwork = DQNet(state_size, action_size, learning_rate, name="DQNetwork")

    # Instantiate the target network
    TargetNetwork = DQNet(state_size, action_size, learning_rate, name="TargetNetwork")

    # Setup TensorBoard Writer
    writer = tf.summary.FileWriter("/tensorboard/dqn/1")

    ## Losses
    tf.summary.scalar("Loss", DQNetwork.loss)

    write_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.summary.FileWriter('log' + '/train', sess.graph)

        # Initialize the variables
        sess.run(tf.global_variables_initializer())

        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0

        # Set tau = 0
        tau = 0

        # Init the game
        # game.init()

        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(total_episodes):
            # Set step to 0
            step = 0

            # Initialize the rewards of the episode
            episode_rewards = []

            # Make a new episode and observe the first state
            state = env.reset()
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stack_frames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # Increase the C step
                tau += 1

                # Increase decay_step
                decay_step += 1

                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability = predict_action(sess, DQNetwork, explore_start, explore_stop, decay_rate,
                                                             decay_step, state, possible_actions)
                encoded_actions = possible_actions[action]

                # Do the action
                next_state, reward, done, _ = env.step(action)

                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros((110, 84), dtype=np.int)
                        next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                        # Set step = max_steps to end the episode
                        step = max_steps

                        # Get the total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        # Check whether the model is better than the best performing one until now
                        if total_reward > highest_reward:
                            save_path = saver.save(sess, "./per_model.ckpt")
                            print("Model saved")
                            highest_reward = total_reward

                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))

                        # Add experience to memory, first one hot encoding it!
                        experience = state, encoded_actions, reward, next_state, done
                        memory.store(experience)

                else:
                    # Stack the frame of the next_state
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

                    # Add experience to memory
                    experience = state, encoded_actions, reward, next_state, done
                    memory.store(experience)

                    # st+1 is now our current state
                    state = next_state

                ### LEARNING PART
                # Obtain random mini-batch from memory
                batch_size = 16
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)
                # print('Samples ISWeights')
                # print(ISWeights_mb)

                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                target_Qs_batch = []

                # DQN with PER
                # action values for all possible actions from the TARGET network

                # Get Q values for next_state
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output, feed_dict={TargetNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a') 
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                # TODO: double check here !!!
                # print('Actions mb before encoding', len(actions_mb), actions_mb)
                # for a in actions_mb:
                # print(a)
                actions_mb_new = []

                # one-hot-encoding of the target actions
                for element in actions_mb:
                    if ((type(element) is np.int64) or (type(element) is int)):
                        empty_list = np.zeros((action_size))
                        empty_list[element] = 1
                        actions_mb_new.append(empty_list)
                    else:
                        actions_mb_new.append(element)
                        # print('ele type', type(element))

                actions_mb_new = np.array(actions_mb_new)

                # print('Actions', actions_mb_new.shape, type(actions_mb_new),  actions_mb_new)
                # print('ISweights', ISweights_mb_new.shape, type(ISWeights_mb), ISWeights_mb)
                ISweights_mb_new = []

                # one-hot-encoding of the target actions

                for element in ISWeights_mb:
                    ISweights_mb_new.append(list(np.array(element)))

                ISweights_mb_new = np.array(ISweights_mb_new)

                # print(len(ISweights_mb_new))
                # print('ISweights_mb_new', type(ISweights_mb_new), ISweights_mb_new.shape, ISweights_mb_new)

                # ISWeights_mb = list(ISWeights_mb)
                # need the loss for every batch in order to update its value in memory

                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                                    # DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors
                                                    feed_dict={DQNetwork.inputs_: states_mb,
                                                               DQNetwork.target_Q: targets_mb,
                                                               DQNetwork.actions_: actions_mb_new,
                                                               DQNetwork.ISWeights_: ISweights_mb_new})
                # Update priority
                memory.batch_update(tree_idx, absolute_errors)

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb_new,
                                                        DQNetwork.ISWeights_: ISweights_mb_new})
                writer.add_summary(summary, episode)
                writer.flush()
        print('Training completed!')


if training == True:
    if sampling_strategy == 'PER':
        print('Training with Prioritized Experience Replay...')
        train_per(stacked_frames)
    else:
        print('Training basic DQN...')
        train_simple(stacked_frames)
