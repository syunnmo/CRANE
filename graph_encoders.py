import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph_encoder1(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(Graph_encoder1, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W1 = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.reduceDim = nn.Linear(in_features * 2, self.out_features, bias=True)
        self.E = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal(self.W1)
        nn.init.kaiming_normal(self.reduceDim.weight)
        nn.init.constant(self.reduceDim.bias, 0)
        nn.init.kaiming_normal(self.E)
        nn.init.kaiming_normal(self.a)

    def forward(self, exercise_h, kc_h, adj_exercise_kc):
        if self.concat:
            kc_Wh = torch.mm(kc_h, self.W1)
            exercise_Wh = torch.mm(exercise_h, self.W1)
            a_input = self._prepare_attentional_mechanism_input(kc_Wh, exercise_Wh)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj_exercise_kc > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            new_kc_embed = torch.matmul(attention, kc_Wh)

            exercise_Eh = torch.mm(exercise_h, self.E)
            exercises_embedd = torch.cat((new_kc_embed, new_kc_embed.mul(exercise_Eh)), dim=1)
            exercises_embedd = self.reduceDim(exercises_embedd)
            return F.elu(exercises_embedd)

    def _prepare_attentional_mechanism_input(self, kc_Wh, exercise_Wh):
        N_kc = kc_Wh.size()[0]
        N_exercise = exercise_Wh.size()[0]
        Wh_repeated_in_chunks = exercise_Wh.repeat_interleave(N_kc, dim=0)
        Wh_repeated_alternating = kc_Wh.repeat(N_exercise, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # 返回的结果是N_exercise*N_kc*2 * self.out_features,前N行即第一个节点和其他所有节点的拼接（包括本节点的拼接）
        return all_combinations_matrix.view(N_exercise, N_kc, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.use_bias = use_bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

    def forward(self, embeddings, adj):
        suport = torch.mm(embeddings, self.weight)
        output = torch.spmm(adj, suport)
        if self.use_bias:
            return output + self.bias
        else:
            return output


class Graph_encoder2(nn.Module):
    def __init__(self, features):
        super(Graph_encoder2, self).__init__()
        self.gcn1 = GraphConvolution(features, features)
        self.gcn2 = GraphConvolution(features, features)

    def forward(self, X, adj):
        X = F.relu(self.gcn1(X, adj))
        X = self.gcn2(X, adj)

        return X