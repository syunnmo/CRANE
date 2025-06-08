import torch
import torch.nn as nn
import utils as utils
import torch.nn.functional as F
from graph_encoders import Graph_encoder1,Graph_encoder2
from lossFunction import lossOfRepresentationLevel
from knowledge_encoders import KnowledgestateAcquisition
class MODEL(nn.Module):
    def __init__(self, n_exercise, batch_size, exercise_embed_dim, hidden_dim, layer_num, interval, params, student_num=None):
        super(MODEL, self).__init__()
        self.n_exercise = n_exercise
        self.n_kc = params.n_knowledge_concept
        self.batch_size = batch_size
        self.exercise_embed_dim = exercise_embed_dim
        self.embed_dim = exercise_embed_dim

        self.student_num = student_num
        self.nheads = params.num_heads
        self.alpha = params.alpha
        self.dropout = params.dropout
        self.gpu = params.gpu
        self.params = params
        self.temperature = params.temperature

        self.hidden_dim = hidden_dim
        self.layer_num = layer_num

        self.prep_embedd = preprocessing_embedd(self.n_exercise, self.exercise_embed_dim, self.nheads, self.dropout, self.alpha, self.gpu)
        self.KEncoder = KnowledgestateAcquisition(self.exercise_embed_dim, self.hidden_dim, interval, self.dropout, self.gpu)
        # self.lambda1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lambda1 = params.lambda1

        self.exercise_embed = nn.Parameter(torch.randn(self.n_exercise, self.exercise_embed_dim))
        self.kc_embed = nn.Parameter(torch.randn(self.n_kc, self.exercise_embed_dim))

        self.fc1 = nn.Linear(self.exercise_embed_dim + self.hidden_dim, self.hidden_dim, bias=None)
        self.fc2 = nn.Linear(self.hidden_dim, 1, bias=None)
        self.rel = nn.ReLU()

        self.init_params()

    def init_params(self):
        nn.init.kaiming_normal(self.fc1.weight)
        nn.init.kaiming_normal(self.fc2.weight)
        nn.init.kaiming_normal(self.exercise_embed)
        nn.init.kaiming_normal(self.kc_embed)

    def forward(self, adj_exercise_kc, adj_EE_view, adj_KK_view, kc_data, exercise_data, exercise_respond_data, target):
        # self.lambda1.data.clamp_(0, 0.5)

        batch_size, seqlen = exercise_data.shape[0], exercise_data.shape[1]

        exercise_node_embedding = self.exercise_embed
        kc_node_mebedding = self.kc_embed

        # Pre-process for exercise with respond data
        slice_respond_embedd_data, slice_exercise_embedd_data, exercise_embedding, contrastive_exercises, contrastive_KCs = \
            self.prep_embedd(exercise_node_embedding, kc_node_mebedding, adj_exercise_kc, adj_EE_view, adj_KK_view, exercise_data, exercise_respond_data, seqlen)

        interactions = torch.cat([slice_respond_embedd_data[i].unsqueeze(1) for i in range(seqlen)], 1)
        assessment_exercises = torch.cat([slice_exercise_embedd_data[i].unsqueeze(1) for i in range(seqlen)], 1)

        # Knowledge states from knowledge encoder
        KStates_KE = self.KEncoder(interactions)

        # Prediction process
        KStates_KE_ = KStates_KE[:, : seqlen - 1, :]
        assessment_exercises_ = assessment_exercises[:, 1:, :]
        y = self.rel(self.fc1(torch.cat((KStates_KE_, assessment_exercises_), -1)))
        pred = self.fc2(y)
        pred = pred.squeeze(-1).view(batch_size * (seqlen - 1), -1)

        target_1d = target                   # [batch_size * seq_len, 1]
        mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        predict_loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        representation_level_loss = lossOfRepresentationLevel(adj_exercise_kc, adj_EE_view,
                                                                               adj_KK_view,
                                                                               exercise_embedding, kc_node_mebedding,
                                                                               contrastive_exercises, contrastive_KCs,
                                                                               self.temperature, self.params.gpu)
        loss = predict_loss + self.lambda1 * representation_level_loss

        return loss, torch.sigmoid(filtered_pred), filtered_target, exercise_embedding, kc_node_mebedding

class preprocessing_embedd(nn.Module):
    def __init__(self, n_exercise, exercise_embed_dim, nheads, dropout, alpha, gpu):
        super(preprocessing_embedd, self).__init__()
        self.n_exercise = n_exercise
        self.exercise_embed_dim = exercise_embed_dim
        self.nheads = nheads
        self.dropout = dropout
        self.alpha = alpha
        self.gpu = gpu

        # Graph_encoder1 for propagation
        self.graph_encoder1 = [
            Graph_encoder1(self.exercise_embed_dim, self.exercise_embed_dim, dropout=self.dropout,
                           alpha=self.alpha, concat=True) for _ in range(self.nheads)]
        for i, attention in enumerate(self.graph_encoder1):
            self.add_module('exercise_kc_attention_{}'.format(i), attention)

        # Graph_encoder2 for propagation
        self.graph_encoder2_EE = Graph_encoder2(self.exercise_embed_dim)
        self.graph_encoder2_KK = Graph_encoder2(self.exercise_embed_dim)

    def forward(self, exercise_node_embedding, kc_node_mebedding, adj_exercise_kc, adj_EE_view, adj_KK_view, exercise_data, exercise_respond_data, seqlen):
        # Generate embedding under different views
        contrastive_exercises = self.graph_encoder2_EE(exercise_node_embedding, adj_EE_view)
        contrastive_KCs = self.graph_encoder2_KK(kc_node_mebedding, adj_KK_view)
        exercise_embedding = torch.cat(
            [att(exercise_node_embedding, kc_node_mebedding, adj_exercise_kc) for att in
             self.graph_encoder1],
            dim=1).view(self.n_exercise, self.exercise_embed_dim, self.nheads).mean(2)

        # Add index 0
        exercise_embedding_add_zero = torch.cat(
            [utils.varible(torch.zeros(1, exercise_embedding.shape[1]), self.gpu),
             exercise_embedding], dim=0)

        slice_exercise_data = torch.chunk(exercise_data, seqlen, 1)
        slice_exercise_embedd_data = []
        for i, single_slice_exercise_data_index in enumerate(slice_exercise_data):
            single_slice_exercise_embedd_data = torch.index_select(exercise_embedding_add_zero, 0,
                                                                   single_slice_exercise_data_index.squeeze(1))
            slice_exercise_embedd_data.append(single_slice_exercise_embedd_data)

        slice_exercise_respond_data = torch.chunk(exercise_respond_data, seqlen, 1)

        # Correct answer is concated the right side of exercise embedding;
        # Incorrect answer is concated the left side of exercise embedding.
        zeros = torch.zeros_like(exercise_embedding)
        cat1 = torch.cat((zeros, exercise_embedding), -1)
        cat2 = torch.cat((exercise_embedding, zeros), -1)
        response_embedding = torch.cat((cat1, cat2), -2)
        response_embedding_add_zero = torch.cat(
            [utils.varible(torch.zeros(1, response_embedding.shape[1]), self.gpu), response_embedding],
            dim=0)
        slice_respond_embedd_data = []
        for i, single_slice_respond_data_index in enumerate(slice_exercise_respond_data):
            single_slice_respond_embedd_data = torch.index_select(response_embedding_add_zero, 0,
                                                                  single_slice_respond_data_index.squeeze(1))
            slice_respond_embedd_data.append(single_slice_respond_embedd_data)

        return slice_respond_embedd_data, slice_exercise_embedd_data, exercise_embedding, contrastive_exercises, contrastive_KCs


