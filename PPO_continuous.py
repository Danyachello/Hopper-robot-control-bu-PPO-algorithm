import torch
from hopper_env_forw import HopperEnvForw
import torch.nn as nn
from torch.distributions import MultivariateNormal
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import imageio


parser = argparse.ArgumentParser(description='PyTorch PPO for continuous controlling')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--env', type=str, default='Hopper', help='continuous env')
parser.add_argument('--render', default=False, action='store_true', help='Render?')
parser.add_argument('--solved_reward', type=float, default= 5000, help='stop training if avg_reward > solved_reward')
parser.add_argument('--print_interval', type=int, default=10, help='how many episodes to print the results out')
parser.add_argument('--save_interval', type=int, default=100, help='how many episodes to save a checkpoint')
parser.add_argument('--max_episodes', type=int, default=100000)
parser.add_argument('--max_timesteps', type=int, default=1500)
parser.add_argument('--update_timesteps', type=int, default=4000, help='how many timesteps to update the policy')
parser.add_argument('--action_std', type=float, default=0.5, help='constant std for action distribution (Multivariate Normal)')
parser.add_argument('--K_epochs', type=int, default=80, help='update the policy for how long time everytime')
parser.add_argument('--eps_clip', type=float, default=0.2, help='epsilon for p/q clipped')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--ckpt_folder', default='./checkpoints', help='Location to save checkpoint models')
parser.add_argument('--log_folder', default='./logs', help='Location to save logs')
parser.add_argument('--mode', default='test', help='choose train or test')
parser.add_argument('--restore', default=False, action='store_true', help='Restore and go on training?')
opt = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:   # collected from old policy
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        #Entrope H_full
        self.action_var = torch.full((action_dim, ), action_std * action_std).to(device)    

    def act(self, state, memory):       
        action_mean = self.actor(state)                     
        cov_mat = torch.diag(self.action_var).to(device)    
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()                             
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):      
        state_value = self.critic(state)    

        
        action_mean = self.actor(state)                     
        action_var = self.action_var.expand_as(action_mean) 
        cov_mat = torch.diag_embed(action_var).to(device)   
        dist = MultivariateNormal(action_mean, cov_mat)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy() #entopy H calculation

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=False, ckpt=None):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # current policy
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        if restore:
            pretained_model = torch.load(ckpt, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretained_model)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        # old policy: initialize old policy with current policy's parameter
        self.old_policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)  # flatten the state
        return self.old_policy.act(state, memory).cpu().numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimation of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        # Normalize rewards
        rewards = torch.FloatTensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(memory.states).to(device)).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device)).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs)).to(device).detach()

        # Train policy for K epochs: sampling and updating
        for _ in range(self.K_epochs):
            # Evaluate old actions and values using current policy
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Importance ratio: p_t(Q)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Advantages A_t = 
            advantages = rewards - state_values.detach()

            # Actor loss using Surrogate loss L_clip
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = - torch.min(surr1, surr2)

            # Critic loss: critic loss - entropy -vL_V + hH
            critic_loss = 0.5 * self.MSE_loss(rewards, state_values) - 0.01 * dist_entropy

            # Total loss -L_clip - vL_v + hH
            loss = actor_loss + critic_loss

            # Backward gradients
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights to old_policy
        self.old_policy.load_state_dict(self.policy.state_dict())


def train(env_name, env, state_dim, action_dim, render, solved_reward,
    max_episodes, max_timesteps, update_timestep, action_std, K_epochs, eps_clip,
    gamma, lr, betas, ckpt_folder, restore, print_interval=10, save_interval=100):

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    if restore:
        print('Load checkpoint from {}'.format(ckpt))

    memory = Memory()

    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=restore, ckpt=ckpt)

    running_reward, avg_length, time_step = 0, 0, 0
    reward_buf = []
    buff = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        state = state[0]
        for t in range(max_timesteps):
            time_step += 1

            # Run old policy
            action = ppo.select_action(state, memory)

            state, reward, done, truncated, info = env.step(action)

            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            if time_step % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0

            running_reward += reward

            buff += reward
            
                

            if render:
                 image = env.render()
                 cv2.imshow("image", image)
                 cv2.waitKey(1)

            if (done or truncated) :
                
                break

        avg_length += t
        reward_buf.append(buff)
        buff = 0
       

        if running_reward > (print_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))

            fig = plt.figure()
            plt.clf()

            plt.plot(range(0,i_episode-1), reward_buf[0:i_episode-1])
            plt.xlabel('episodes')
            plt.ylabel('reward')
            plt.title('episode={}'.format(i_episode))
            plt.grid()

            fig.savefig('media/train.png')

            reward_buf = []

            print('Save a checkpoint!')
            break

        if i_episode % save_interval == 0:
            torch.save(ppo.policy.state_dict(), ckpt_folder + '/PPO_continuous_{}.pth'.format(env_name))
            print('Save a checkpoint!')

        if i_episode % print_interval == 0:
            avg_length = int(avg_length / print_interval)
            running_reward = int((running_reward / print_interval))

            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))

            running_reward, avg_length = 0, 0

def test(env_name, env, state_dim, action_dim, render, action_std, K_epochs, eps_clip, gamma, lr, betas, ckpt_folder, test_episodes):

    ckpt = ckpt_folder+'/PPO_continuous_'+env_name+'.pth'
    print('Load checkpoint from {}'.format(ckpt))

    memory = Memory()

    ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip, restore=True, ckpt=ckpt)

    episode_reward, time_step = 0, 0
    avg_episode_reward, avg_length = 0, 0
    frames = []
    ctrl_buf = []
    ctrl_buf_2 = []

    # test
    for i_episode in range(1, test_episodes+1):
        state = env.reset()
        state = state[0]
        buf = 0
        while True:
            time_step += 1

            # Run old policy
            action = ppo.select_action(state, memory)

            state, reward, done, truncated, info = env.step(action)
            ctrl_buf.append(info["upper_motor"])
            ctrl_buf_2.append(info["lower_motor"])

            episode_reward += reward

            if render:
                image = env.render()
                if time_step % 5 == 0:
                    frames.append(image)
                cv2.imshow("image", image)
                cv2.waitKey(1)

            if (done or truncated):
                print('Episode {} \t Length: {} \t Reward: {}'.format(i_episode, time_step, episode_reward))
                avg_episode_reward += episode_reward
                avg_length += time_step
                memory.clear_memory()
                with imageio.get_writer("media/test_{}.gif".format(i_episode), mode="I") as writer:
                    for idx, frame in enumerate(frames):
                        writer.append_data(frame)

                fig = plt.figure()
                plt.clf()

                plt.plot(range(0, time_step), ctrl_buf[0:time_step], label = "upper_motor" )
                plt.plot(range(0, time_step), ctrl_buf_2[0:time_step],label = "lower_motor")
                plt.legend()
                plt.xlabel('timesteps')
                plt.ylabel('torques')
                plt.title('Applied torques')
                plt.grid()

                fig.savefig('media/torques_{}.png'.format(i_episode))
                time_step, episode_reward = 0, 0
                ctrl_buf_2 =[]
                ctrl_buf = []
                frames = []

                break

    print('Test {} episodes DONE!'.format(test_episodes))
    print('Avg episode reward: {} | Avg length: {}'.format(avg_episode_reward/test_episodes, avg_length/test_episodes))


if __name__ == '__main__':

    if not os.path.exists(opt.ckpt_folder):
        os.mkdir(opt.ckpt_folder)

    env_name = opt.env
    env = HopperEnvForw(render_mode="rgb_array")
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print('Environment: {}\nState Size: {}\nAction Size: {}\n'.format(env_name, state_dim, action_dim))

    if opt.mode == 'train':
        train(env_name, env, state_dim, action_dim,
            render=opt.render, solved_reward=opt.solved_reward,
            max_episodes=opt.max_episodes, max_timesteps=opt.max_timesteps, update_timestep=opt.update_timesteps,
            action_std=opt.action_std, K_epochs=opt.K_epochs, eps_clip=opt.eps_clip,
            gamma=opt.gamma, lr=opt.lr, betas=[0.9, 0.990], ckpt_folder=opt.ckpt_folder,
            restore=opt.restore, print_interval=opt.print_interval, save_interval=opt.save_interval)
    elif opt.mode == 'test':
        test(env_name, env, state_dim, action_dim,
            render=opt.render, action_std=opt.action_std, K_epochs=opt.K_epochs, eps_clip=opt.eps_clip,
            gamma=opt.gamma, lr=opt.lr, betas=[0.9, 0.990], ckpt_folder=opt.ckpt_folder, test_episodes=5)
    else:
        raise Exception("Wrong Mode!")
