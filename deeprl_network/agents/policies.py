import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.utils import batch_to_seq, init_layer, one_hot, run_rnn
from pdb import set_trace as stx

class Policy(nn.Module):
    def __init__(self, n_a, n_s, n_step, policy_name, agent_name, identical):
        super(Policy, self).__init__()
        self.name = policy_name
        if agent_name is not None:
            # for multi-agent system
            self.name += '_' + str(agent_name)
        self.n_a = n_a
        self.n_s = n_s
        self.n_step = n_step
        self.identical = identical

    def forward(self, ob, *_args, **_kwargs):
        raise NotImplementedError()

    def _init_actor_head(self, n_h, n_a=None):
        if n_a is None:
            n_a = self.n_a
        # only discrete control is supported for now
        self.actor_head = nn.Linear(n_h, n_a)
        init_layer(self.actor_head, 'fc')

    def _init_critic_head(self, n_h, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            if self.identical:
                n_na_sparse = self.n_a*n_n
            else:
                n_na_sparse = sum(self.na_dim_ls)
            n_h += n_na_sparse
        self.critic_head = nn.Linear(n_h, 1)
        init_layer(self.critic_head, 'fc')

    def _run_critic_head(self, h, na, n_n=None):
        if n_n is None:
            n_n = int(self.n_n)
        if n_n:
            na = torch.from_numpy(na).long()
            if self.identical:
                na_sparse = one_hot(na, self.n_a)
                na_sparse = na_sparse.view(-1, self.n_a*n_n)
            else:
                na_sparse = []
                na_ls = torch.chunk(na, n_n, dim=1)
                for na_val, na_dim in zip(na_ls, self.na_dim_ls):
                    na_sparse.append(torch.squeeze(one_hot(na_val, na_dim), dim=1))
                na_sparse = torch.cat(na_sparse, dim=1)
            h = torch.cat([h, na_sparse], dim=1)
        return self.critic_head(h).squeeze()

    def _run_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs): # here
        # stx()
        log_probs = actor_dist.log_prob(As)
        policy_loss = -(log_probs * Advs).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef
        return policy_loss, value_loss, entropy_loss

    def _run_ppo_loss(self, actor_dist, e_coef, v_coef, vs, As, Rs, Advs, old_log_prob): # here
        ppo_clip_high = 0.2; ppo_clip_low = 0.2

        log_prob = actor_dist.log_prob(As)
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio * Advs
        clip_ratio = torch.clamp(ratio, 1.0 - ppo_clip_low, 1.0 + ppo_clip_high)
        surr2 = clip_ratio * Advs
        clip_frac = (ratio != clip_ratio).float().mean().item()
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -(actor_dist.entropy()).mean() * e_coef
        value_loss = (Rs - vs).pow(2).mean() * v_coef

        return policy_loss, value_loss, entropy_loss, clip_frac

    def _update_tensorboard(self, summary_writer, global_step):
        # monitor training
        summary_writer.add_scalar('loss/{}_entropy_loss'.format(self.name), self.entropy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_policy_loss'.format(self.name), self.policy_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_value_loss'.format(self.name), self.value_loss,
                                  global_step=global_step)
        summary_writer.add_scalar('loss/{}_total_loss'.format(self.name), self.loss,
                                  global_step=global_step)
        if hasattr(self, 'clip_frac'):
            summary_writer.add_scalar(f'item/clip_frac', self.clip_frac, global_step=global_step)


class LstmPolicy(Policy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(LstmPolicy, self).__init__(n_a, n_s, n_step, 'lstm', name, identical)
        if not self.identical:
            self.na_dim_ls = na_dim_ls
        self.n_lstm = n_lstm
        self.n_fc = n_fc
        self.n_n = n_n
        self._init_net()
        self._reset()

    def backward(self, obs, nactions, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float()
        dones = torch.from_numpy(dones).float()
        xs = self._encode_ob(obs)
        hs, new_states = run_rnn(self.lstm_layer, xs, dones, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        actor_dist = torch.distributions.categorical.Categorical(logits=F.log_softmax(self.actor_head(hs), dim=1)) # 旧版本写法 ?? 不影响结果。因为log_softmax(log_softmax(logits)) = log_softmax(logits)
        # actor_dist = torch.distributions.categorical.Categorical(logits=self.actor_head(hs))
        vs = self._run_critic_head(hs, nactions)
        self.policy_loss, self.value_loss, self.entropy_loss = \
            self._run_loss(actor_dist, e_coef, v_coef, vs,
                           torch.from_numpy(acts).long(),
                           torch.from_numpy(Rs).float(),
                           torch.from_numpy(Advs).float())
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, naction=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float()
        x = self._encode_ob(ob)
        h, new_states = run_rnn(self.lstm_layer, x, done, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return F.softmax(self.actor_head(h), dim=1).squeeze().detach().numpy()
        else:
            return self._run_critic_head(h, np.array([naction])).detach().numpy()

    def _encode_ob(self, ob):
        return F.relu(self.fc_layer(ob))

    def _init_net(self):
        self.fc_layer = nn.Linear(self.n_s, self.n_fc)
        init_layer(self.fc_layer, 'fc')
        self.lstm_layer = nn.LSTMCell(self.n_fc, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _reset(self):
        # forget the cumulative states every cum_step
        self.states_fw = torch.zeros(self.n_lstm * 2)
        self.states_bw = torch.zeros(self.n_lstm * 2)


class FPPolicy(LstmPolicy):
    def __init__(self, n_s, n_a, n_n, n_step, n_fc=64, n_lstm=64, name=None,
                 na_dim_ls=None, identical=True):
        super(FPPolicy, self).__init__(n_s, n_a, n_n, n_step, n_fc, n_lstm, name,
                         na_dim_ls, identical)

    def _init_net(self):
        if self.identical:
            self.n_x = self.n_s - self.n_n * self.n_a
        else:
            self.n_x = int(self.n_s - sum(self.na_dim_ls))
        self.fc_x_layer = nn.Linear(self.n_x, self.n_fc)
        init_layer(self.fc_x_layer, 'fc')
        n_h = self.n_fc
        if self.n_n:
            self.fc_p_layer = nn.Linear(self.n_s-self.n_x, self.n_fc)
            init_layer(self.fc_p_layer, 'fc')
            n_h += self.n_fc
        self.lstm_layer = nn.LSTMCell(n_h, self.n_lstm)
        init_layer(self.lstm_layer, 'lstm')
        self._init_actor_head(self.n_lstm)
        self._init_critic_head(self.n_lstm)

    def _encode_ob(self, ob):
        x = F.relu(self.fc_x_layer(ob[:, :self.n_x]))
        if self.n_n:
            p = F.relu(self.fc_p_layer(ob[:, self.n_x:]))
            x = torch.cat([x, p], dim=1)
        return x


class NCMultiAgentPolicy(Policy):
    """ Inplemented as a centralized meta-DNN. To simplify the implementation, all input
    and output dimensions are identical among all agents, and invalid values are casted as
    zeros during runtime."""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        super(NCMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'nc', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float()
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()
        hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
        # backward grad is limited to the minibatch
        self.states_bw = new_states.detach()
        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts) # [25, 120, 64], [25, 120] -> 25,120
        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()
        if summary_writer:
            summary_writer.add_scalar(f'item/adv', Advs.mean(), global_step=global_step)
        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                    acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i
        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)
            
    def ppo_backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, optimizer, summary_writer=None, global_step=None):                
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        dones = torch.from_numpy(dones).float()
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()

        with torch.no_grad():
            hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
            ps = self._run_actor_heads(hs)
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float() # frozen advantages
        if summary_writer:
            summary_writer.add_scalar(f'item/adv', Advs.mean(), global_step=global_step)

        Advs = (Advs - Advs.mean()) / (Advs.std() + 1e-8)
        old_log_probs = [torch.distributions.categorical.Categorical(logits=ps[i]).log_prob(acts[i]) for i in range(self.n_agent)]

        ppo_steps = 4
        for _ in range(ppo_steps):
            optimizer.zero_grad()
            hs, new_states = self._run_comm_layers(obs, dones, fps, self.states_bw)
            if _ == ppo_steps - 1:
                self.states_bw = new_states.detach()
            vs = self._run_critic_heads(hs, acts)
            ps_new = self._run_actor_heads(hs)
            
            self.policy_loss = 0
            self.value_loss = 0
            self.entropy_loss = 0
            self.clip_frac = 0
            for i in range(self.n_agent):
                actor_dict = torch.distributions.categorical.Categorical(logits=ps_new[i])
                policy_loss_i, value_loss_i, entropy_loss_i, clip_frac = \
                    self._run_ppo_loss(actor_dict, e_coef, v_coef, vs[i], acts[i], Rs[i], Advs[i], old_log_probs[i])
                self.policy_loss += policy_loss_i
                self.value_loss += value_loss_i
                self.entropy_loss += entropy_loss_i
                self.clip_frac += clip_frac
            self.loss = self.policy_loss + self.value_loss + self.entropy_loss
            self.loss.backward()

            nn.utils.clip_grad_norm_(self.parameters(), 40)
            optimizer.step()

        self.clip_frac /= ppo_steps * self.n_agent
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        # BxNxm
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float() # [1,25,12]
        done = torch.from_numpy(np.expand_dims(done, axis=0)).float() # [1]
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float() # [1,25,5]
        # h dim: NxBxm
        h, new_states = self._run_comm_layers(ob, done, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True) # h:[25, 1, 64] action:[25, 1] -> 25,1

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        p_i = torch.index_select(p, 0, js)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            p_i = p_i.view(1, self.n_a * n_n)
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            p_i_ls = []
            nx_i_ls = []
            for j in range(n_n):
                p_i_ls.append(p_i[j].narrow(0, 0, self.na_ls_ls[i][j]))
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            p_i = torch.cat(p_i_ls).unsqueeze(0)
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        s_i = [F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))), # 自身观测 + 邻居观测
               F.relu(self.fc_p_layers[i](p_i)), # 邻居指纹
               F.relu(self.fc_m_layers[i](m_i))] # 邻居隐藏状态（消息）
        return torch.cat(s_i, dim=1)

    def _get_neighbor_dim(self, i_agent):
        n_n = int(np.sum(self.neighbor_mask[i_agent]))
        if self.identical:
            return n_n, self.n_s * (n_n+1), self.n_a * n_n, [self.n_s] * n_n, [self.n_a] * n_n
        else:
            ns_ls = []
            na_ls = []
            for j in np.where(self.neighbor_mask[i_agent])[0]:
                ns_ls.append(self.n_s_ls[j])
                na_ls.append(self.n_a_ls[j])
            return n_n, self.n_s_ls[i_agent] + sum(ns_ls), sum(na_ls), ns_ls, na_ls

    def _init_actor_head(self, n_a):
        # only discrete control is supported for now
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, n_n, n_ns, n_na):
        n_lstm_in = 3 * self.n_fc
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_p_layer = nn.Linear(n_na, self.n_fc)
            init_layer(fc_p_layer, 'fc')
            fc_m_layer = nn.Linear(self.n_h * n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
            self.fc_p_layers.append(fc_p_layer)
            lstm_layer = nn.LSTMCell(n_lstm_in, self.n_h)
        else:
            self.fc_m_layers.append(None)
            self.fc_p_layers.append(None)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_na):
        critic_head = nn.Linear(self.n_h + n_na, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.fc_p_layers = nn.ModuleList()
        self.fc_m_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.ns_ls_ls = []
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, n_ns, n_na, ns_ls, na_ls = self._get_neighbor_dim(i)
            self.ns_ls_ls.append(ns_ls)
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            self._init_comm_layer(n_n, n_ns, n_na)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().detach().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1) # ??
            ps.append(p_i)
        return ps

    def _run_comm_layers(self, obs, dones, fps, states):
        # (B,25,12) (B) (B,25,5) (25,128)
        obs = batch_to_seq(obs)
        dones = batch_to_seq(dones)
        fps = batch_to_seq(fps)
        h, c = torch.chunk(states, 2, dim=1)
        outputs = []
        for t, (x, p, done) in enumerate(zip(obs, fps, dones)):
            next_h = []
            next_c = []
            x = x.squeeze(0)
            p = p.squeeze(0)
            for i in range(self.n_agent):
                n_n = self.n_n_ls[i]
                if n_n:
                    s_i = self._get_comm_s(i, n_n, x, h, p) # s_i.shape = [1,64*(n_n+1)]. 0,2,[25,12],[25,64],[25,5]
                else:
                    if self.identical:
                        x_i = x[i].unsqueeze(0)
                    else:
                        x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
                    s_i = F.relu(self.fc_x_layers[i](x_i))
                h_i, c_i = h[i].unsqueeze(0) * (1-done), c[i].unsqueeze(0) * (1-done)
                next_h_i, next_c_i = self.lstm_layers[i](s_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)
            h, c = torch.cat(next_h), torch.cat(next_c)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        for i in range(self.n_agent):
            n_n = self.n_n_ls[i]
            if n_n:
                js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
                na_i = torch.index_select(actions, 0, js)
                na_i_ls = []
                for j in range(n_n):
                    na_i_ls.append(one_hot(na_i[j], self.na_ls_ls[i][j]))
                h_i = torch.cat([hs[i]] + na_i_ls, dim=1)
            else:
                h_i = hs[i]
            v_i = self.critic_heads[i](h_i).squeeze()
            if detach:
                vs.append(v_i.detach().numpy())
            else:
                vs.append(v_i)
        return vs
  

class ConsensusPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cu', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def consensus_update(self):
        consensus_update = []
        with torch.no_grad():
            for i in range(self.n_agent):
                mean_wts = self._get_critic_wts(i)
                for param, wt in zip(self.lstm_layers[i].parameters(), mean_wts):
                    param.copy_(wt)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()
        self.na_ls_ls = []
        self.n_n_ls = []
        for i in range(self.n_agent):
            n_n, _, n_na, _, na_ls = self._get_neighbor_dim(i)
            n_s = self.n_s if self.identical else self.n_s_ls[i]
            self.na_ls_ls.append(na_ls)
            self.n_n_ls.append(n_n)
            fc_x_layer = nn.Linear(n_s, self.n_fc)
            init_layer(fc_x_layer, 'fc')
            self.fc_x_layers.append(fc_x_layer)
            lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
            init_layer(lstm_layer, 'lstm')
            self.lstm_layers.append(lstm_layer)
            n_a = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a)
            self._init_critic_head(n_na)

    def _get_critic_wts(self, i_agent):
        wts = []
        for wt in self.lstm_layers[i_agent].parameters():
            wts.append(wt.detach())
        neighbors = list(np.where(self.neighbor_mask[i_agent] == 1)[0])
        for j in neighbors:
            for k, wt in enumerate(self.lstm_layers[j].parameters()):
                wts[k] += wt.detach()
        n = 1 + len(neighbors)
        for k in range(len(wts)):
            wts[k] /= n
        return wts

    def _run_comm_layers(self, obs, dones, fps, states):
        # NxTxm
        obs = obs.transpose(0, 1)
        hs = []
        new_states = []
        for i in range(self.n_agent):
            xs_i = F.relu(self.fc_x_layers[i](obs[i]))
            hs_i, new_states_i = run_rnn(self.lstm_layers[i], xs_i, dones, states[i])
            hs.append(hs_i.unsqueeze(0))
            new_states.append(new_states_i.unsqueeze(0))
        return torch.cat(hs), torch.cat(new_states)


class CommNetMultiAgentPolicy(NCMultiAgentPolicy):
    """Reference code: https://github.com/IC3Net/IC3Net/blob/master/comm.py.
       Note in CommNet, the message is generated from hidden state only, so current state
       and neigbor policies are not included in the inputs."""
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'cnet', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        m_i = torch.index_select(h, 0, js).mean(dim=0, keepdim=True)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
            x_i = x[i].unsqueeze(0)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
            x_i = x[i].narrow(0, 0, self.n_s_ls[i]).unsqueeze(0)
        return F.relu(self.fc_x_layers[i](torch.cat([x_i, nx_i], dim=1))) + \
               self.fc_m_layers[i](m_i)


class DIALMultiAgentPolicy(NCMultiAgentPolicy):
    def __init__(self, n_s, n_a, n_agent, n_step, neighbor_mask, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        Policy.__init__(self, n_a, n_s, n_step, 'dial', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls
        self.n_agent = n_agent
        self.neighbor_mask = neighbor_mask
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def _init_comm_layer(self, n_n, n_ns, n_na):
        fc_x_layer = nn.Linear(n_ns, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)
        if n_n:
            fc_m_layer = nn.Linear(self.n_h*n_n, self.n_fc)
            init_layer(fc_m_layer, 'fc')
            self.fc_m_layers.append(fc_m_layer)
        else:
            self.fc_m_layers.append(None)
        lstm_layer = nn.LSTMCell(self.n_fc, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _get_comm_s(self, i, n_n, x, h, p):
        js = torch.from_numpy(np.where(self.neighbor_mask[i])[0]).long()
        m_i = torch.index_select(h, 0, js).view(1, self.n_h * n_n)
        nx_i = torch.index_select(x, 0, js)
        if self.identical:
            nx_i = nx_i.view(1, self.n_s * n_n)
        else:
            nx_i_ls = []
            for j in range(n_n):
                nx_i_ls.append(nx_i[j].narrow(0, 0, self.ns_ls_ls[i][j]))
            nx_i = torch.cat(nx_i_ls).unsqueeze(0)
        a_i = one_hot(p[i].argmax().unsqueeze(0), self.n_fc)
        return F.relu(self.fc_x_layers[i](torch.cat([x[i].unsqueeze(0), nx_i], dim=1))) + \
               F.relu(self.fc_m_layers[i](m_i)) + a_i


class IC3NetMultiAgentPolicy(Policy):
    """
    IC3Net Multi-Agent Policy.
    This implementation focuses on global information and a gating mechanism,
    removing direct neighbor-to-neighbor communication.
    """
    def __init__(self, n_s, n_a, n_agent, n_step, n_fc=64, n_h=64,
                 n_s_ls=None, n_a_ls=None, identical=True):
        super(IC3NetMultiAgentPolicy, self).__init__(n_a, n_s, n_step, 'ic3net', None, identical)
        if not self.identical:
            self.n_s_ls = n_s_ls
            self.n_a_ls = n_a_ls

        self.n_agent = n_agent
        self.n_fc = n_fc
        self.n_h = n_h
        self._init_net()
        self._reset()

    def backward(self, obs, fps, acts, dones, Rs, Advs,
                 e_coef, v_coef, summary_writer=None, global_step=None):
        obs = torch.from_numpy(obs).float().transpose(0, 1)
        fps = torch.from_numpy(fps).float().transpose(0, 1)
        acts = torch.from_numpy(acts).long()
        
        hs, new_states = self._run_comm_layers(obs, fps, self.states_bw)
        self.states_bw = new_states.detach()

        ps = self._run_actor_heads(hs)
        vs = self._run_critic_heads(hs, acts)

        self.policy_loss = 0
        self.value_loss = 0
        self.entropy_loss = 0
        Rs = torch.from_numpy(Rs).float()
        Advs = torch.from_numpy(Advs).float()

        if summary_writer:
            summary_writer.add_scalar(f'item/adv', Advs.mean(), global_step=global_step)

        for i in range(self.n_agent):
            actor_dist_i = torch.distributions.categorical.Categorical(logits=ps[i])
            policy_loss_i, value_loss_i, entropy_loss_i = \
                self._run_loss(actor_dist_i, e_coef, v_coef, vs[i],
                               acts[i], Rs[i], Advs[i])
            self.policy_loss += policy_loss_i
            self.value_loss += value_loss_i
            self.entropy_loss += entropy_loss_i

        self.loss = self.policy_loss + self.value_loss + self.entropy_loss
        self.loss.backward()
        if summary_writer is not None:
            self._update_tensorboard(summary_writer, global_step)

    def forward(self, ob, done, fp, action=None, out_type='p'):
        ob = torch.from_numpy(np.expand_dims(ob, axis=0)).float()
        fp = torch.from_numpy(np.expand_dims(fp, axis=0)).float()

        h, new_states = self._run_comm_layers(ob, fp, self.states_fw)
        if out_type.startswith('p'):
            self.states_fw = new_states.detach()
            return self._run_actor_heads(h, detach=True)
        else:
            action = torch.from_numpy(np.expand_dims(action, axis=1)).long()
            return self._run_critic_heads(h, action, detach=True)

    def _get_comm_input_dim(self, i_agent):
        if self.identical:
            obs_input_dim = self.n_s
        else:
            obs_input_dim = self.n_s_ls[i_agent]

        lstm_input_dim = self.n_fc + self.n_h
        return obs_input_dim, lstm_input_dim

    def _init_actor_head(self, n_a):
        actor_head = nn.Linear(self.n_h, n_a)
        init_layer(actor_head, 'fc')
        self.actor_heads.append(actor_head)

    def _init_comm_layer(self, obs_input_dim, lstm_input_dim):
        fc_x_layer = nn.Linear(obs_input_dim, self.n_fc)
        init_layer(fc_x_layer, 'fc')
        self.fc_x_layers.append(fc_x_layer)

        # Communication Gate for the *incoming* global message (remains the same)
        gate_layer = nn.Linear(self.n_h, 1)
        init_layer(gate_layer, 'fc')
        self.gate_layers.append(gate_layer)

        lstm_layer = nn.LSTMCell(lstm_input_dim, self.n_h)
        init_layer(lstm_layer, 'lstm')
        self.lstm_layers.append(lstm_layer)

    def _init_critic_head(self, n_a_agent):
        critic_head = nn.Linear(self.n_h + n_a_agent, 1)
        init_layer(critic_head, 'fc')
        self.critic_heads.append(critic_head)

    def _init_net(self):
        self.fc_x_layers = nn.ModuleList()
        self.gate_layers = nn.ModuleList() # Gate for *incoming* global message
        self.lstm_layers = nn.ModuleList()
        self.actor_heads = nn.ModuleList()
        self.critic_heads = nn.ModuleList()

        # NEW: Gate layers for *outgoing* information contribution to global pool
        self.contrib_gate_layers = nn.ModuleList()

        # Calculate initial total contribution dim for a single agent: n_h + n_a
        # This assumes identical agents for this calculation; if not, you'd iterate n_a_ls[i]
        total_contrib_dim_per_agent = self.n_h + self.n_a

        # Global Message Layer: aggregates *gated contributions* from all agents
        # Input: N_agents * (n_h + n_a)
        self.global_message_layer = nn.Linear(self.n_agent * total_contrib_dim_per_agent, self.n_h)
        init_layer(self.global_message_layer, 'fc')

        for i in range(self.n_agent):
            obs_input_dim, lstm_input_dim = self._get_comm_input_dim(i)
            self._init_comm_layer(obs_input_dim, lstm_input_dim)

            # Initialize gate for outgoing contribution for agent i
            # Input to this gate is the concatenated (fingerprint + h)
            contrib_gate_layer = nn.Linear(total_contrib_dim_per_agent, 1)
            init_layer(contrib_gate_layer, 'fc')
            self.contrib_gate_layers.append(contrib_gate_layer)

            n_a_agent = self.n_a if self.identical else self.n_a_ls[i]
            self._init_actor_head(n_a_agent)
            self._init_critic_head(n_a_agent)

    def _reset(self):
        self.states_fw = torch.zeros(self.n_agent, self.n_h * 2)
        self.states_bw = torch.zeros(self.n_agent, self.n_h * 2)

    def _run_actor_heads(self, hs, detach=False):
        ps = []
        for i in range(self.n_agent):
            if detach:
                p_i = F.softmax(self.actor_heads[i](hs[i]), dim=1).squeeze().detach().numpy()
            else:
                p_i = F.log_softmax(self.actor_heads[i](hs[i]), dim=1)
            ps.append(p_i)
        return ps

    def _run_comm_layers(self, obs, fps, states):
        h, c = torch.chunk(states, 2, dim=1) # [25,64]

        outputs = []
        for t in range(obs.shape[0]):
            x_t = obs[t].squeeze(0)
            p_t = fps[t].squeeze(0) # Fingerprint at current time step (action from prev time step)

            next_h = []
            next_c = []

            # --- NEW: Collect gated contributions from all agents for global message ---
            gated_agent_contributions = []
            for i in range(self.n_agent):
                # Concatenate current fingerprint (p_t[i]) and hidden state (h[i])
                # Ensure fingerprint is properly shaped if it's not already (1, n_a)
                # p_t[i] is (n_a), h[i] is (n_h)
                # Assuming p_t[i] is already one-hot or appropriate (e.g. from previous action)
                
                # If fingerprints are not one-hot, make sure they are before concatenation
                # Or ensure self.n_a matches the actual dimension of the raw fingerprint
                combined_info = torch.cat([p_t[i], h[i]], dim=0).unsqueeze(0) # (1, n_a + n_h)


                # Get the gate output for this agent's contribution to the global pool
                contrib_gate_output = torch.sigmoid(self.contrib_gate_layers[i](combined_info)) # (1, 1) ??

                # Apply the gate to the combined info
                gated_contrib = combined_info * contrib_gate_output # (1, n_a + n_h)
                gated_agent_contributions.append(gated_contrib)

            # Concatenate all gated contributions to form the input for global_message_layer
            all_gated_contributions = torch.cat(gated_agent_contributions, dim=1) # (1, N_agents * (n_a + n_h))

            # Generate the global message from these gated contributions
            global_message = F.relu(self.global_message_layer(all_gated_contributions)) # (1, n_h) ??

            # --- Original IC3Net logic for processing individual agent inputs ---
            for i in range(self.n_agent):
                x_i = x_t[i].unsqueeze(0)

                fc_x_output = F.relu(self.fc_x_layers[i](x_i))

                # Gate for incoming global message (remains the same)
                gate_output = torch.sigmoid(self.gate_layers[i](h[i].unsqueeze(0))) # 从全局信息中获取信息时使用的门 ??
                gated_global_message = global_message * gate_output

                lstm_input_i = torch.cat([fc_x_output, gated_global_message], dim=1)

                h_i, c_i = h[i].unsqueeze(0), c[i].unsqueeze(0)
                next_h_i, next_c_i = self.lstm_layers[i](lstm_input_i, (h_i, c_i))
                next_h.append(next_h_i)
                next_c.append(next_c_i)

            h, c = torch.cat(next_h), torch.cat(next_c)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs)
        return outputs.transpose(0, 1), torch.cat([h, c], dim=1)

    def _run_critic_heads(self, hs, actions, detach=False):
        vs = []
        for i in range(self.n_agent):
            h_i_agent = hs[i]
            actions_i_agent = actions[i]
            n_a_agent = self.n_a if self.identical else self.n_a_ls[i]
            agent_actions_one_hot = one_hot(actions_i_agent, n_a_agent)

            h_i_critic_input = torch.cat([h_i_agent, agent_actions_one_hot], dim=1)

            v_i = self.critic_heads[i](h_i_critic_input).squeeze()

            if detach:
                vs.append(v_i.detach().numpy())
            else:
                vs.append(v_i)
        return vs