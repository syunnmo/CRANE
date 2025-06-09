import torch
import torch.nn as nn
import numpy as np
import utils as utils
from torch.autograd import Variable
import torch.nn.functional as F

class KEncoderWithLSEM(nn.Module):
    def __init__(self, exercise_embed_dim, hidden_dim, layer_num, dropout, gpu):
        super(KEncoderWithLSEM, self).__init__()
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(exercise_embed_dim * 2, hidden_dim, layer_num, batch_first=True, dropout = dropout)
        self.gpu = gpu
    def forward(self, interactions):
        h0 = utils.varible(Variable(torch.zeros(self.layer_num, interactions.size(0), self.hidden_dim)), self.gpu)
        c0 = utils.varible(Variable(torch.zeros(self.layer_num, interactions.size(0), self.hidden_dim)), self.gpu)
        knowledge_state, (hn, cn) = self.lstm(interactions, (h0, c0))
        return knowledge_state


# LSTM with learning efficency
class CustomLSTMwithLE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMwithLE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_g = nn.Linear(input_size, hidden_size)

        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.U_g = nn.Linear(hidden_size, hidden_size)

        self.W_h = nn.Linear(hidden_size * 2, hidden_size)
        self.init_params()
    def init_params(self):
        nn.init.kaiming_normal(self.W_i.weight)
        nn.init.kaiming_normal(self.W_f.weight)
        nn.init.kaiming_normal(self.W_o.weight)
        nn.init.kaiming_normal(self.W_g.weight)
        nn.init.kaiming_normal(self.U_i.weight)
        nn.init.kaiming_normal(self.U_f.weight)
        nn.init.kaiming_normal(self.U_o.weight)
        nn.init.kaiming_normal(self.U_g.weight)
        nn.init.kaiming_normal(self.W_h.weight)
    def forward(self, x, prev_h, prev_c, curr_le):
        # x means the current interaction;
        i_t = torch.sigmoid(self.W_i(x) + self.U_i(prev_h))
        f_t = torch.sigmoid(self.W_f(x) + self.U_f(prev_h))
        o_t = torch.sigmoid(self.W_o(x) + self.U_o(prev_h))
        g_t = torch.tanh(self.W_g(x) + self.U_g(prev_h))
        c_t = f_t * prev_c + i_t * g_t
        h_t_tilde = o_t * torch.tanh(c_t)
        return h_t_tilde, c_t
class KnowledgestateAcquisition(nn.Module):
    def __init__(self, exercise_embed_dim, hidden_dim, interval, dropout, gpu):
        super(KnowledgestateAcquisition, self).__init__()
        self.exercise_embed_dim = exercise_embed_dim
        self.hidden_dim = hidden_dim
        self.interval = interval
        self.gpu = gpu
        self.lstmCell = nn.LSTMCell(input_size=self.exercise_embed_dim * 2, hidden_size=self.hidden_dim)
        self.W_h = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, interactions):
        batch_size, len = interactions.shape[0], interactions.shape[1]

        h0 = utils.varible(Variable(torch.zeros(batch_size, self.hidden_dim)), self.gpu)
        c0 = utils.varible(Variable(torch.zeros(batch_size, self.hidden_dim)), self.gpu)
        le0 = utils.varible(torch.nn.Parameter(torch.randn(batch_size, self.hidden_dim), requires_grad=True), self.gpu)
        gain0 = utils.varible(torch.nn.Parameter(torch.randn(batch_size), requires_grad=True), self.gpu)
        gain0.data.clamp_(0.001, 0.999)
        assert (
                len % self.interval == 0
        ), "interaction length needs to be divisible by interval"

        h_list =[]; c_list = []; LE_list = []; gain_list = []
        h_list.append(h0); c_list.append(c0); LE_list.append(le0); gain_list.append(gain0)
        for i in range(len):
            prev_h = h_list[-1]
            prev_c = c_list[-1]
            prev_gain = gain_list[-1]
            prev_le = LE_list[-1]
            # update learning efficiency
            if i % self.interval == 0 and i != 0:
                prev_prev_h = h_list[-(self.interval)]
                gain = (1 - F.cosine_similarity(prev_prev_h, prev_h, dim=1)) / (self.interval)
                curr_le = (1 + (gain - prev_gain)).unsqueeze(1) * prev_le
                gain_list.append(gain); LE_list.append(curr_le)
            else:
                curr_le = prev_le
            current_interaction = torch.squeeze(interactions[:, i : i + 1, ])
            h_t_tilde, c_t = self.lstmCell(current_interaction, (prev_h, prev_c))
            h_t_tilde_ = self.dropout(h_t_tilde)
            h_t = self.W_h(torch.cat((h_t_tilde_, curr_le), dim=1))
            h_list.append(h_t)
            c_list.append(c_t)
        del h_list[0]
        h = torch.cat(h_list, dim=-1).view(batch_size, len, self.hidden_dim)
        return h