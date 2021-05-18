from copy import deepcopy
import argparse
import heapq
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import gym
import gym.spaces
import numpy as np
import core
import sparse_gym_mujoco
from ES import sepCEM, Control
from models import RLNN
from random_process import GaussianNoise, OrnsteinUhlenbeckProcess
from memory import Memory
from util import *
import os
gpuid=4
print(gpuid)
torch.cuda.set_device(gpuid)
cur_path = os.path.abspath(os.curdir)
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    FloatTensor = torch.cuda.FloatTensor
else:
    FloatTensor = torch.FloatTensor
EvaluationReturn = 'remote_evaluation/Average Returns'
n1 = s1 = s2 = 0
mmd_states = []
Qeval = 'q_critic_estimation'
Qmc = 'q_MC_real'
Qerr = 'q_error'
Epoch = 'Epoch'
test_num = 0
Qevaluations = []
def better(scorelist, oldscore):
    betterid_list = []
    for i in range(len(scorelist)):
        if scorelist[i] > oldscore:
            betterid_list.append(i)
    return betterid_list
def evaluate(actor, env, memory=None, n_episodes=1, random=False, noise=None, render=False):
    """
    Computes the score of an actor on a given number of runs,
    fills the memory if needed
    """

    def policy(state):
        state = FloatTensor(state.reshape(-1))
        action = actor(state).cpu().data.numpy().flatten()
        return np.clip(action, -max_action, max_action)
    scores = []
    steps = 0

    for _ in range(n_episodes):

        score = 0
        obs = deepcopy(env.reset())
        done = False

        while not done:

            # get next action and act
            action = policy(obs)
            n_obs, reward, done, _ = env.step(action)
            done_bool = 0 if steps + \
                1 == env._max_episode_steps else float(done)
            score += reward
            steps += 1


            if memory is not None:
                memory.add((obs, n_obs, action, reward, done_bool))
            obs = n_obs
            if done:
                env.reset()

        scores.append(score)

    #return np.mean(scores), steps
    return scores, steps
def evaluateq(actor, env, eval_episodes=None,critic = None):
    """
    Computes the score of an actor on a given number of runs,
    calculates the error of between evaluate_q and true_q
    calculates true_q using Monte Carlo
    """
    global test_num
    def policy(state):
        state = FloatTensor(state.reshape(-1))
        action = actor(state).cpu().data.numpy().flatten()
        return np.clip(action, -max_action, max_action)
    scores = []
    for _ in range(eval_episodes):
        steps = 0
        score = 0
        obs = deepcopy(env.reset())
        
        action = policy(obs)
        obs = FloatTensor(obs)
        action = FloatTensor(action)
        action = torch.unsqueeze(action, 0)
        obs = torch.unsqueeze(obs, 0)
        qeval = critic(obs,action)[0]
        qeval = torch.squeeze(qeval, 0)
        qeval = qeval.cpu().data.numpy()[0]
        rewList = []
        
        done = False
        while not (done or (steps == env._max_episode_steps)):
            # get next action and act
            action = policy(obs)
            n_obs, reward, done, _ = env.step(action)
            rewList.append(reward)
            score += reward
            steps += 1
            obs = n_obs
            if done:
                env.reset()
        rew_array = np.array(rewList)
        qmc = core.discount_cumsum(rew_array, 0.99)[0]
        scores.append(score)
        qerr = qeval - qmc
        Qevaluations.append([qeval,qmc,qerr,score,test_num])
        test_num += 1
        if  (test_num+1)>980:
            test = pd.DataFrame(columns=[Qeval,Qmc,Qerr, EvaluationReturn, Epoch], data=Qevaluations)
            test.to_csv(goal_path + '/' + 'progress.csv')
    return scores

class Actor(RLNN):

    def __init__(self, state_dim, action_dim, max_action, args):
        super(Actor, self).__init__(state_dim, action_dim, max_action)

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)
        self.layer_norm = args.layer_norm

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x):

        if not self.layer_norm:
            x = torch.tanh(self.l1(x))
            x = torch.tanh(self.l2(x))
            x = self.max_action * torch.tanh(self.l3(x))

        else:
            x = torch.tanh(self.n1(self.l1(x)))
            x = torch.tanh(self.n2(self.l2(x)))
            x = self.max_action * torch.tanh(self.l3(x))

        return x
    # MMD functions
    def compute_gau_kernel(self, x, y, sigma):
        batch_size = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        dim = x.shape[2]
        tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
        tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
        return torch.exp(-(tiled_x - tiled_y).pow(2).sum(dim=3) / (2 * sigma))

    # MMD functions
    def compute_lap_kernel(self, x, y, sigma):
        batch_size = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        dim = x.shape[2]
        tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
        tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
        return torch.exp(-torch.abs(tiled_x - tiled_y).sum(dim=3) / sigma)

    def compute_mmd(self, x, y, kernel='lap'):
        if kernel == 'gau':
            x_kernel = self.compute_gau_kernel(x, x, 20)
            y_kernel = self.compute_gau_kernel(y, y, 20)
            xy_kernel = self.compute_gau_kernel(x, y, 20)
        else:
            x_kernel = self.compute_lap_kernel(x, x, 10)
            y_kernel = self.compute_lap_kernel(y, y, 10)
            xy_kernel = self.compute_lap_kernel(x, y, 10)
        square_mmd = x_kernel.mean((1, 2)) + y_kernel.mean((1, 2)) - 2 * xy_kernel.mean((1, 2))
        return square_mmd
    def mmdupdate(self,memory,batch_size,actor_best,critic, actor_t,n=None):
        with torch.no_grad():
            states, _, _, _, _ = memory.sample(batch_size)
            states = states.cpu().data.numpy()
            state_rep_m = torch.FloatTensor(np.repeat(states, n, axis=0)).cuda()
            state_rep_n = torch.FloatTensor(np.repeat(states, n, axis=0)).cuda()
            actions_m = actor_best(state_rep_m)
            actions_m = actions_m.view(batch_size, n, -1)
        actions_n = self(state_rep_n)
        actions_n = actions_n.view(batch_size, n, -1)
        mmd_dist = self.compute_mmd(actions_m, actions_n)
        mmd_loss = mmd_dist.mean()
        states, _, _, _, _ = memory.sample(batch_size)
        actor_loss = -critic(states, self(states))[0].mean()+0.2*mmd_loss
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
    def update(self, memory, batch_size, critic, actor_t):

        # Sample replay buffer
        states, _, _, _, _ = memory.sample(batch_size)

        # Compute actor loss
        if args.use_td3:
            actor_loss = -critic(states, self(states))[0].mean()
        else:
            actor_loss = -critic(states, self(states)).mean()

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), actor_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class Critic(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(Critic, self).__init__(state_dim, action_dim, 1)

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

    def forward(self, x, u):

        if not self.layer_norm:
            x = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x = F.leaky_relu(self.l2(x))
            x = self.l3(x)

        else:
            x = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x = F.leaky_relu(self.n2(self.l2(x)))
            x = self.l3(x)

        return x

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Q target = reward + discount * Q(next_state, pi(next_state))
        with torch.no_grad():
            target_Q = critic_t(n_states, actor_t(n_states))
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimate
        current_Q = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)


class CriticTD3(RLNN):
    def __init__(self, state_dim, action_dim, max_action, args):
        super(CriticTD3, self).__init__(state_dim, action_dim, 1)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n1 = nn.LayerNorm(400)
            self.n2 = nn.LayerNorm(300)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 400)
        self.l5 = nn.Linear(400, 300)
        self.l6 = nn.Linear(300, 1)

        if args.layer_norm:
            self.n4 = nn.LayerNorm(400)
            self.n5 = nn.LayerNorm(300)

        self.layer_norm = args.layer_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_lr)
        self.tau = args.tau
        self.discount = args.discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

    def forward(self, x, u):

        if not self.layer_norm:
            x1 = F.leaky_relu(self.l1(torch.cat([x, u], 1)))
            x1 = F.leaky_relu(self.l2(x1))
            x1 = self.l3(x1)

        else:
            x1 = F.leaky_relu(self.n1(self.l1(torch.cat([x, u], 1))))
            x1 = F.leaky_relu(self.n2(self.l2(x1)))
            x1 = self.l3(x1)

        if not self.layer_norm:
            x2 = F.leaky_relu(self.l4(torch.cat([x, u], 1)))
            x2 = F.leaky_relu(self.l5(x2))
            x2 = self.l6(x2)

        else:
            x2 = F.leaky_relu(self.n4(self.l4(torch.cat([x, u], 1))))
            x2 = F.leaky_relu(self.n5(self.l5(x2)))
            x2 = self.l6(x2)

        return x1, x2

    def update(self, memory, batch_size, actor_t, critic_t):

        # Sample replay buffer
        states, n_states, actions, rewards, dones = memory.sample(batch_size)

        # Select action according to policy and add clipped noise
        noise = np.clip(np.random.normal(0, self.policy_noise, size=(
            batch_size, action_dim)), -self.noise_clip, self.noise_clip)
        n_actions = actor_t(n_states) + FloatTensor(noise)
        n_actions = n_actions.clamp(-max_action, max_action)

        # Q target = reward + discount * min_i(Qi(next_state, pi(next_state)))
        with torch.no_grad():
            target_Q1, target_Q2 = critic_t(n_states, n_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - dones) * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + \
            nn.MSELoss()(current_Q2, target_Q)

        # Optimize the critic
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.parameters(), critic_t.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)
def mmd_policy(state):
        action = actor(state).cpu().data.numpy()
        return np.clip(action, -max_action, max_action)
# MMD functions
def compute_gau_kernel1( x, y, sigma):
    batch_size = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    dim = x.shape[2]
    tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
    tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
    return torch.exp(-(tiled_x - tiled_y).pow(2).sum(dim=3) / (2 * sigma))


# MMD functions
def compute_lap_kernel1( x, y, sigma):
    batch_size = x.shape[0]
    x_size = x.shape[1]
    y_size = y.shape[1]
    dim = x.shape[2]
    tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
    tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
    return torch.exp(-torch.abs(tiled_x - tiled_y).sum(dim=3) / sigma)


def compute_mmd1(x, y, kernel='lap'):
    if kernel == 'gau':
        x_kernel = compute_gau_kernel1(x, x, 20)
        y_kernel = compute_gau_kernel1(y, y, 20)
        xy_kernel = compute_gau_kernel1(x, y, 20)
    else:
        x_kernel =compute_lap_kernel1(x, x, 10)
        y_kernel = compute_lap_kernel1(y, y, 10)
        xy_kernel = compute_lap_kernel1(x, y, 10)
    square_mmd = x_kernel.mean((1, 2)) + y_kernel.mean((1, 2)) - 2 * xy_kernel.mean((1, 2))
    return square_mmd.mean()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='train', type=str,)#Walker2d
    parser.add_argument('--env', default='SparseWalker2d-v1', type=str)#HalfCheetah
    parser.add_argument('--seed', default=121, type=int)
    parser.add_argument('--start_steps', default=10000, type=int)

    # DDPG parameters
    parser.add_argument('--actor_lr', default=0.001, type=float)
    parser.add_argument('--critic_lr', default=0.001, type=float)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--mmd_batch_size', default=600, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--reward_scale', default=1., type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--layer_norm', dest='layer_norm', action='store_true')

    # TD3 parameters
    parser.add_argument('--use_td3', default=True, type=float)
    parser.add_argument('--policy_noise', default=0.2, type=float)
    parser.add_argument('--mmd_sigma', default=0.2, type=float)
    parser.add_argument('--noise_clip', default=0.5, type=float)
    parser.add_argument('--policy_freq', default=2, type=int)

    # Gaussian noise parameters
    parser.add_argument('--gauss_sigma', default=0.1, type=float)

    # OU process parameters
    parser.add_argument('--ou_noise', dest='ou_noise', action='store_true')
    parser.add_argument('--ou_theta', default=0.15, type=float)
    parser.add_argument('--ou_sigma', default=0.2, type=float)
    parser.add_argument('--ou_mu', default=0.0, type=float)

    # ES parameters
    parser.add_argument('--pop_size', default=10, type=int)
    parser.add_argument('--elitism', dest="elitism",  action='store_true')
    parser.add_argument('--n_grad', default=5, type=int)
    parser.add_argument('--sigma_init', default=1e-3, type=float)
    parser.add_argument('--damp', default=1e-3, type=float)
    parser.add_argument('--damp_limit', default=1e-5, type=float)
    parser.add_argument('--mult_noise', dest='mult_noise', action='store_true')

    # Training parameters
    parser.add_argument('--n_episodes', default=1, type=int)
    parser.add_argument('--max_steps', default=1000001, type=int)
    parser.add_argument('--mem_size', default=1000000, type=int)
    parser.add_argument('--n_noisy', default=0, type=int)

    # Testing parameters
    parser.add_argument('--filename', default="", type=str)
    parser.add_argument('--n_test', default=1, type=int)

    # misc
    parser.add_argument('--output', default='results/', type=str)
    parser.add_argument('--period', default=10000, type=int)
    parser.add_argument('--n_eval', default=10, type=int)
    parser.add_argument('--save_all_models',
                        dest="save_all_models", action="store_true")
    parser.add_argument('--debug', dest='debug', action='store_true')

    parser.add_argument('--render', dest='render', action='store_true')

    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    with open(args.output + "/parameters.txt", 'w') as file:
        for key, value in vars(args).items():
            file.write("{} = {}\n".format(key, value))

    # environment
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = int(env.action_space.high[0])
    print(env)
    print("seed =",args.seed)
    # memory
    memory = Memory(args.mem_size, state_dim, action_dim)

    # critic
    if args.use_td3:
        critic = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t = CriticTD3(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    else:
        critic = Critic(state_dim, action_dim, max_action, args)
        critic_t = Critic(state_dim, action_dim, max_action, args)
        critic_t.load_state_dict(critic.state_dict())

    # actor
    actor = Actor(state_dim, action_dim, max_action, args)
    actor_t = Actor(state_dim, action_dim, max_action, args)
    actor_t.load_state_dict(actor.state_dict())
    actor_best = Actor(state_dim, action_dim, max_action, args)
    # action noise
    if not args.ou_noise:
        a_noise = GaussianNoise(action_dim, sigma=args.gauss_sigma)
    else:
        a_noise = OrnsteinUhlenbeckProcess(
            action_dim, mu=args.ou_mu, theta=args.ou_theta, sigma=args.ou_sigma)

    if USE_CUDA:
        critic.cuda()
        critic_t.cuda()
        actor.cuda()
        actor_t.cuda()
        actor_best.cuda()
    goal_path = cur_path + '/' + str(args.env) + '/' + 'seed_' + str(args.seed)
    os.makedirs(goal_path)
    # CEM
    es = sepCEM(actor.get_size(), mu_init=actor.get_params(), sigma_init=args.sigma_init, damp=args.damp, damp_limit=args.damp_limit,
                pop_size=args.pop_size, antithetic=not args.pop_size % 2, parents=args.pop_size // 2, elitism=args.elitism)
    # es = Control(actor.get_size(), pop_size=args.pop_size, mu_init=actor.get_params())
    m=n=5
    start =0
    n1 = s1 = s2 =0
    mmd_states=[]
    evaluationss=best_mmd_action=[]
    oldbest_score = 0
    step_cpt = 0
    total_steps = 0
    actor_steps = 0
    df = pd.DataFrame(columns=["total_steps", "average_score",
                               "average_score_rl", "average_score_ea", "best_score"])
    while total_steps < args.max_steps:

        fitness = []
        fitness_ = []
        es_params = es.ask(args.pop_size)

        # udpate the rl actors and the critic
        if total_steps > args.start_steps:

            for i in range(args.n_grad):
                fitness = []
                # set params
                actor.set_params(es_params[i])
                actor_t.set_params(es_params[i])
                actor.optimizer = torch.optim.Adam(
                    actor.parameters(), lr=args.actor_lr)

                for _ in (range(actor_steps // args.n_grad)):
                          critic.update(memory, args.batch_size, actor, critic_t)
                # actor update
                if(i < 4):
                    for _ in (range(actor_steps)):
                        actor.update(memory, args.batch_size,
                                     critic, actor_t)
                else:
                    for _ in (range(actor_steps)):
                          actor.mmdupdate(memory, args.batch_size,
                                  actor_best,critic, actor_t,n=5)

                # get the params back in the population
                es_params[i] = actor.get_params()
        actor_steps = 0


        # evaluate all actors
        for params in es_params:

            actor.set_params(params)
            f, steps = evaluate(actor, env, memory=memory, n_episodes=args.n_episodes,
                                render=args.render)
            f=np.mean(f)
            actor_steps += steps
            fitness.append(f)

            # print scores
            #prLightPurple('Actor fitness:{}'.format(f))

        mmd_dist_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        if  start:
            old_best_mmd_states = mmd_states
        if max(fitness)>oldbest_score and start:
            betterid = better(fitness,oldbest_score)
            best_id = fitness.index(max(fitness))
            #actor_best.set_params(es_params[best_id])
            for j in betterid:
                actor.set_params(es_params[j])
                mmd_action = mmd_policy(old_best_mmd_states)
                mmd_action = torch.FloatTensor(mmd_action).cuda()
                mmd_action = mmd_action.view(args.mmd_batch_size, n, -1)
                mmd_dist = compute_mmd1(mmd_action, old_best_mmd_action)
                mmd_dist = mmd_dist.cpu().data.numpy()
                mmd_dist_list[j]=mmd_dist
        # update es
        scoresfitness = np.array(fitness)
        scoresfitness *= -1
        idx_sorted = np.argsort(scoresfitness)
        if max(mmd_dist_list) > 0:
            max_mmd_id = mmd_dist_list.index(max(mmd_dist_list))
            actor_best.set_params(es_params[max_mmd_id])

            scores_mmd_dist_list = np.array(mmd_dist_list)
            rank_mmd = np.arange(len(scores_mmd_dist_list))
            rank_mmds = rank_mmd[scores_mmd_dist_list > 0]
            le = len(rank_mmds)
            if le > 1:
                scores_mmd_dist_list *= -1
                idx_sorted1 = np.argsort(scores_mmd_dist_list)
                arr3 = idx_sorted[0]
                index = np.argwhere(idx_sorted1 == arr3)
                a1 = np.delete(idx_sorted1, index)
                arr1 = a1[:le-1]
                #arr1 = arr1[::-1]
                arr2 = idx_sorted[le:]
                idx_sorted2 = np.append(arr3, arr1)
                idx_sorted = np.append(idx_sorted2,arr2)
        # update es
        es.tell(es_params, fitness,idx_sorted)

        # update step counts
        total_steps += actor_steps
        step_cpt += actor_steps

        # save stuff
        if step_cpt >= args.period:
            s1 = total_steps // 1000
            s3 = s1 - s2
            start = 1
            # evaluate mean actor over several runs. Memory is not filled
            # and steps are not counted
            actor.set_params(es.mu)
            with torch.no_grad():
                mmd_states, _, _, _, _ = memory.sample(args.mmd_batch_size)
            mmd_states = mmd_states.cpu().data.numpy()
            mmd_states = torch.FloatTensor(np.repeat(mmd_states, n, axis=0)).cuda()
            best_mmd_action = mmd_policy(mmd_states)
            best_mmd_action = torch.FloatTensor(best_mmd_action).cuda()
            best_mmd_action = best_mmd_action.view(args.mmd_batch_size, n, -1)
            old_best_mmd_action = best_mmd_action
            f_list = evaluateq(actor, env,eval_episodes=s3, critic = critic)
            f_mu = np.mean(f_list)
            step_cpt = 0
            s2 = s1
            oldbest_score = f_mu
