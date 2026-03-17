import os
import numpy as np
import click
import json, time
import torch
import random
import multiprocessing as mp
from itertools import product
import glob, ast

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics

import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from numpy.random import default_rng
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer

from rlkit.torch.encoder import RNNEncoder, MLPEncoder
from rlkit.torch.decoder import FOCALDecoder

import gym


rng = default_rng()

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# metric loss in FOCAL
def metric_loss(z, tasks, epsilon=1e-3):
    # z shape is (task, corresponding dim)
    pos_z_loss = 0.
    neg_z_loss = 0.
    pos_cnt = 0
    neg_cnt = 0
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            # positive pair
            if tasks[i] == tasks[j]:
                pos_z_loss += torch.sqrt(torch.mean((z[i] - z[j]) ** 2) + epsilon)
                pos_cnt += 1
            else:
                neg_z_loss += 1/(torch.mean((z[i] - z[j]) ** 2) + epsilon * 100)  
                neg_cnt += 1
    #print(pos_z_loss, pos_cnt, neg_z_loss, neg_cnt)
    return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)


def expand_variant_with_env(variant, include_act_space=False):
    env_params = variant.get("env_params", {})
    util_params = variant.get("util_params", {})

    # env = make_env(
    #     env_name=variant["env_name"],
    #     max_rollouts_per_task=variant["max_rollouts_per_task"],
    #     seed=util_params.get("seed", 0),
    #     n_tasks=1
    # )
    
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))

    # ===== action space =====
    if isinstance(env.action_space, gym.spaces.Discrete):
        variant["action_dim"] = 1
        variant["act_space_n"] = env.action_space.n
    else:
        variant["action_dim"] = env.action_space.shape[0]
        variant["act_space_n"] = None

    # ===== observation =====
    variant["obs_dim"] = env.observation_space.shape[0]

    # ===== trajectory =====
    variant["trajectory_len"] = (
        env.unwrapped._max_episode_steps
        * variant["max_rollouts_per_task"]
    )

    # ===== optional =====
    variant["num_states"] = (
        env.unwrapped.num_states
        if hasattr(env.unwrapped, "num_states")
        else None
    )

    if include_act_space:
        variant["act_space"] = env.action_space

    return variant, env


def experiment(variant, seed=None):
    # env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    variant, env = expand_variant_with_env(variant)
    
    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()   # range(1,20)

    obs_dim = int(np.prod(env.observation_space.shape))  # 27
    action_dim = int(np.prod(env.action_space.shape))    # 8
    # print('YinCH')
    # print(obs_dim, action_dim)
    # return
    reward_dim = 1
    obs_normalizer = ptu.RunningMeanStd(shape=obs_dim)

    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    net_size = variant['net_size']
    use_next_obs_in_context = variant['algo_params']['use_next_obs_in_context']  # False


    # YinCH_todo: 具体的参数添加没找到位置
    if variant['encoder_type'] == 'rnn':
        encoder = RNNEncoder(
            layers_before_gru=variant['layers_before_aggregator'],
            hidden_size=variant['aggregator_hidden_size'],
            layers_after_gru=variant['layers_after_aggregator'],
            task_embedding_size=variant['task_embedding_size'],
            action_size = variant["act_space_n"] if variant["act_space_n"] is not None else variant["action_dim"], # fixed a bug?
            action_embed_size=variant['action_embedding_size'],
            state_size=variant['obs_dim'],
            state_embed_size=variant['state_embedding_size'],
            reward_size=1,
            reward_embed_size=variant['reward_embedding_size'],
        ).to(ptu.device)
    elif variant['encoder_type'] == 'mlp':
        encoder = MLPEncoder(
                hidden_size=variant['aggregator_hidden_size'],
                num_hidden_layers=2,
                task_embedding_size=variant['task_embedding_size'],
                action_size = variant["act_space_n"] if variant["act_space_n"] is not None else variant["action_dim"],
                state_size=variant['obs_dim'],
                reward_size=1,
                term_size=0,
                stochasity=variant['enc_stochastic']
            ).to(ptu.device)
    else:
        raise NotImplementedError
    
    decoder = FOCALDecoder(
			obs_size=variant['obs_dim'],
			action_size=variant['action_dim'],
			task_embedding_size=variant['task_embedding_size'],
			task_embedd_is_deterministic=False,
			device=ptu.device,
			num_layers=2,
			hidden_size=variant['aggregator_hidden_size'],
			ensemble_size=variant['ensemble_size'],
		).to(ptu.device)
    
    finetune_encoder_decoder_optimizer = torch.optim.Adam([{"params": encoder.parameters()}, {"params": decoder.parameters()}], lr=variant['encoder_lr'])

    def supervised_loss(task_embedding, input_obs, input_action, supervise_next_obs, supervise_reward):

        # task_embedding = task_embedding.repeat(len(input_obs), 1, 1).reshape(-1, variant['task_embedding_size'])
        task_embedding = task_embedding.unsqueeze(0).repeat(input_obs.size(0), 1, 1)
        task_embedding = task_embedding.reshape(-1, variant['task_embedding_size'])
        
        input_obs = input_obs.reshape(-1, variant['obs_dim'])
        input_action = input_action.reshape(-1, variant['action_dim'])
        supervise_next_obs = supervise_next_obs.reshape(-1, variant['obs_dim'])
        supervise_reward = supervise_reward.reshape(-1, 1)
        
        return decoder.loss(task_embedding, input_obs, input_action, supervise_next_obs, supervise_reward)
    
    
    # if use_next_obs_in_context:
    #     task_dynamics =  MultiTaskDynamics(num_tasks=variant['n_train_tasks'], 
    #                                  hidden_size=net_size, 
    #                                  num_hidden_layers=3, 
    #                                  action_dim=action_dim, 
    #                                  obs_dim=obs_dim,
    #                                  reward_dim=1,
    #                                  use_next_obs_in_context=use_next_obs_in_context,
    #                                  ensemble_size=variant['algo_params']['ensemble_size'],
    #                                  dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5])
    # else:
    #     task_dynamics = MultiTaskDynamics(num_tasks=variant['n_train_tasks'], 
    #                                  hidden_size=net_size, 
    #                                  num_hidden_layers=2, 
    #                                  action_dim=action_dim, 
    #                                  obs_dim=obs_dim,
    #                                  reward_dim=1,
    #                                  use_next_obs_in_context=use_next_obs_in_context,
    #                                  ensemble_size=variant['algo_params']['ensemble_size'],
    #                                  dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5])
    train_tasks = list(tasks[:variant['n_train_tasks']]) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_buffer = MultiTaskReplayBuffer(variant['algo_params']['replay_buffer_size'], env, train_tasks, 1)

    train_trj_paths = []
    for n in range(variant['algo_params']['n_trj']):
        train_trj_paths += glob.glob(os.path.join(variant['algo_params']['data_dir'], "goal_idx*", "trj_evalsample%d_step%d.npy" %(n, variant['algo_params']['train_epoch'])))
        # dpath = os.path.join(variant['algo_params']['data_dir'], "goal_idx*", "trj_evalsample%d_step%d.npy" %(n, variant['algo_params']['train_epoch']))
        # print(dpath)
        # print("found traj files:", len(train_trj_paths))
        # print(train_trj_paths[:10])
        # return

        train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in train_tasks]
        train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in train_tasks]
        
        obs_train_lst = []
        action_train_lst = []
        reward_train_lst = []
        next_obs_train_lst = []
        terminal_train_lst = []
        task_train_lst = []

        for train_path, train_task_idx in zip(train_paths, train_task_idxs):
            trj_npy = np.load(train_path, allow_pickle=True)
            obs_train_lst += list(trj_npy[:, 0])
            action_train_lst += list(trj_npy[:, 1])
            reward_train_lst += list(trj_npy[:, 2])
            next_obs_train_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_train_lst += terminal
            task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
            task_train_lst += task_train
        
    obs_normalizer.update(obs_train_lst)
    env.update_obs_mean_var(obs_normalizer.mean, obs_normalizer.var)
    obs_train_lst = obs_normalizer.forward(obs_train_lst)
    next_obs_train_lst = obs_normalizer.forward(next_obs_train_lst)
        
        # load training buffer
    for i, (
            task_train, obs, action, reward, next_obs, terminal,
    ) in enumerate(zip(
            task_train_lst,
            obs_train_lst,
            action_train_lst,
            reward_train_lst,
            next_obs_train_lst,
            terminal_train_lst,
    )):
        train_buffer.add_sample(task_train, obs, action, reward, terminal, next_obs, **{'env_info': {}},)


    encoder.train(True)
    # for task_idx in train_tasks:
    #     data = train_buffer.get_all_data(task_idx)
    #     task_dynamics.set_task_idx(task_idx)
    #     task_dynamics.train(data)
    #     print(f"Task {task_idx} finished training")
    # work_dir = os.getcwd()
    # os.makedirs(work_dir+'/gentle_data/asset/dynamics/'+variant['env_name']+f'/expert_seed{seed}', exist_ok=True)
    # task_dynamics.save(work_dir+'/gentle_data/asset/dynamics/'+variant['env_name']+f'/expert_seed{seed}')

    for step1 in range(variant['num_iters']):
        indices = np.random.choice(train_tasks, size=variant['meta_batch'], replace=True)

        for step2 in range(variant['decoder_iter']):
            # sample corresponding context batch. Here assume to use self.storage to provide context

            # obs_context, actions_context, rewards_context, next_obs_context, terms_context = [], [], [], [], []
            obs_data_list, action_data_list, reward_data_list, next_obs_data_list, terms_data_list= [], [], [], [], []
            for task_id in indices:
                data = train_buffer.get_all_data(task_id)
                obs_data_list.append(torch.from_numpy(data["observations"]).float())
                action_data_list.append(torch.from_numpy(data["actions"]).float())
                reward_data_list.append(torch.from_numpy(data["rewards"]).float())
                next_obs_data_list.append(torch.from_numpy(data["next_observations"]).float())
                terms_data_list.append(torch.from_numpy(data["terminals"]).float())
            min_bs = min(d.size(0) for d in obs_data_list)
            obs_data_list = [d[:min_bs] for d in obs_data_list]
            action_data_list = [d[:min_bs] for d in action_data_list]
            reward_data_list = [d[:min_bs] for d in reward_data_list]
            next_obs_data_list = [d[:min_bs] for d in next_obs_data_list]
            terms_data_list = [d[:min_bs] for d in terms_data_list]
            # obs_context = torch.stack(obs_data_list, axis=0).to(ptu.device)
            # actions_context = torch.stack(action_data_list, axis=0).to(ptu.device)
            # rewards_context = torch.stack(reward_data_list, axis=0).to(ptu.device)
            # next_obs_context = torch.stack(next_obs_data_list, axis=0).to(ptu.device)
            # terms_context = torch.stack(terms_data_list, axis=0).to(ptu.device)
            obs_context = torch.stack(obs_data_list, dim=0).permute(1, 0, 2).to(ptu.device)
            actions_context = torch.stack(action_data_list, dim=0).permute(1, 0, 2).to(ptu.device)
            rewards_context = torch.stack(reward_data_list, dim=0).permute(1, 0, 2).to(ptu.device)
            next_obs_context = torch.stack(next_obs_data_list, dim=0).permute(1, 0, 2).to(ptu.device)
            terms_context = torch.stack(terms_data_list, dim=0).permute(1, 0, 2).to(ptu.device)
            
            
            #update context encoder with contrastive loss   
            task_encoding, encoder_loss = encoder.context_encoding(obs=obs_context, actions=actions_context, 
                rewards=rewards_context, next_obs=next_obs_context, terms=terms_context)
            total_loss = metric_loss(task_encoding, indices) * variant['beta_encoder']
            total_loss += supervised_loss(task_embedding=task_encoding,
                                            input_obs=obs_context,
                                            input_action=actions_context,
                                            supervise_next_obs=next_obs_context,
                                            supervise_reward=rewards_context)
            # total_loss += self.args.beta_encoder * encoder_loss
            
            if step1*step2 % 100 == 0:
                print(
                    f"[Step {step1}.{step2}] "
                    f"total_loss={total_loss.item():.4f}, "
                    f"metric_loss={metric_loss(task_encoding, indices).item():.4f}, "
                    f"supervised_loss={supervised_loss(task_embedding=task_encoding, input_obs=obs_context, input_action=actions_context, supervise_next_obs=next_obs_context, supervise_reward=rewards_context).item():.4f}")

            
            finetune_encoder_decoder_optimizer.zero_grad()
            total_loss.backward()
            finetune_encoder_decoder_optimizer.step()
            
    encoder.train(False)
    work_dir = os.getcwd()
    save_dir = os.path.join(
        work_dir,
        'gentle_data/asset/dynamics',
        variant['env_name'],
        f'expert_seed{seed}'
    )
    os.makedirs(save_dir, exist_ok=True)
    encoder.save(save_dir+'/encoder.pth')
    decoder.save(save_dir+'/decoder.pth')
    print("encoder and decoder saved successfully!")

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--seed_list', default=[0,1,2,3])

def main(config, gpu, seed_list):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    if len(seed_list) > 1:
        if isinstance(seed_list, str):
            seed_list = ast.literal_eval(seed_list)
        p = mp.Pool(len(seed_list))
        p.starmap(experiment, product([variant], seed_list))
    else:
        experiment(variant, seed=seed_list[0])

if __name__ == "__main__":
    main()



