import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
from utils.rl_utils import calculate_quantile_huber_loss
from utils.nvhelp import check_gpu_mem_usedRate
import math
th.autograd.set_detect_anomaly(True)

class ZLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        taus = th.arange(
            0, args.quantiles_num+1, device=self.args.device, dtype=th.float32) / args.quantiles_num
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, args.quantiles_num)

        if args.quantiles_num == 100:
            self.selected_idxs = (th.arange(9, device=args.device) + -4 + args.ucb).clip(0, 99)
        elif args.quantiles_num == 200:
            self.selected_idxs = (th.arange(9, device=args.device) + -4 + args.ucb*2).clip(0, 199)
        else:
            raise ValueError('the number of quantiles must be selected in 100 or 200')


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        th.cuda.empty_cache()
        self.optimiser.zero_grad()
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]   # [b, l, agent, act]
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time  [batch_size, max_seq_length, agent_num, action_num, quantiles]
        # print("mask shape:{}".format(mask.shape))
        # print('convert to batch episode data')
        # print(f"{th.cuda.memory_allocated()/1024**3}gb")
        
        
        # print("mac_out.shape : {}".format(mac_out.shape))
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_zvals = th.gather(mac_out[:, :-1], dim=3, index=actions.unsqueeze(-1).expand(-1,-1,-1,-1,self.args.quantiles_num)).squeeze(3)  # Remove the action dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1).detach()  # Concat across time 

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999 # batch, len, agent, action, quantile

        if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                if self.args.name=='method4':
                    qtls = mac_out_detach[:,:,:,:,self.selected_idxs]
                    mac_out_detach = (1 - self.args.ucb_w) * mac_out_detach.mean(dim=-1) + self.args.ucb_w * qtls.mean(dim=-1)
                else:
                    mac_out_detach = mac_out_detach.mean(dim=-1)
                cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
                target_max_zvals = th.gather(target_mac_out, 3, cur_max_actions.unsqueeze(-1).expand(-1,-1,-1,-1,self.args.quantiles_num)).squeeze(3) # batch, len, agent, quantile
        else:
            raise Exception('unacomplish none double-q')
            target_max_zvals = target_mac_out.max(dim=3)[0]

        agent_num = target_max_zvals.shape[2]
        train_agents_num = min(agent_num,max(3,(int)(agent_num)**0.5))
        for i in range(math.ceil(agent_num/train_agents_num)):
            th.cuda.empty_cache()
            b = i*train_agents_num
            e = min(b+train_agents_num,agent_num)
            idx = th.arange(b,e)
            # Calculate 1-step Q-Learning targets
            cur_target_max_zvals=target_max_zvals[:,:,idx,:].unsqueeze(-1).transpose(3, 4)
            num = idx.shape[0]
            targets = rewards.expand(-1,-1,num) [..., None, None]+ self.args.gamma * (1 - terminated.expand(-1,-1,num)[..., None, None]) * cur_target_max_zvals
            cur_chosen_action_zvals=chosen_action_zvals[:,:,idx,:].unsqueeze(-1)
            # loss = (cur_chosen_action_zvals**2).sum()
            # Td-error
            # print('before td')
            # print(f"{th.cuda.memory_allocated()/1024**3}gb")
            td_error = (cur_chosen_action_zvals - targets.detach())  # batch_size epsidode_len agent_num N N
            cur_mask = mask[...,None,None].expand_as(td_error)

            # 0-out the targets that came from padded data
            masked_td_error = td_error * cur_mask
            th.cuda.empty_cache() 
            # quantile_huber_loss
            loss = calculate_quantile_huber_loss(masked_td_error, self.tau_hats) / cur_mask.sum()
            # Optimise
            loss.backward(retain_graph=True)
            
            
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip).to('cpu')
        self.optimiser.step()
        # print('after step loss')
        # print(f"{th.cuda.memory_allocated()/1024**3}gb")

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            # print(f"chosen_action_zvals shape:{chosen_action_zvals.mean(dim=-1).shape}")
            # print(mask.expand_as(chosen_action_zvals.mean(dim=-1)).shape)
            chosen_action_zvals.mean(dim=-1) * mask.expand_as(chosen_action_zvals.mean(dim=-1))
            self.logger.log_stat("q_taken_mean", (chosen_action_zvals.mean(dim=-1) * mask.expand_as(chosen_action_zvals.mean(dim=-1))).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
