import torch
import torch.nn.functional as F
import utils as utils

def lossOfKnowledgeLevel(KStates_KE1, KStates_KE2, temperature, gpu):
    # KStates_KE1, KStates_KE2: (batch_size, length, embed_dim)
    KStates_KE1_, KStates_KE2_ = F.normalize(KStates_KE1, dim=-1), F.normalize(KStates_KE2, dim=-1)
    batch_size, len, embed_dim = KStates_KE1_.shape[0], KStates_KE1_.shape[1], KStates_KE1_.shape[2]

    # CL_loss of knowledge states
    intra_similarity_matrix = KnowledgeLossOfSimilarityMatrix(KStates_KE1_, KStates_KE1_, temperature)
    inter_similarity_matrix = KnowledgeLossOfSimilarityMatrix(KStates_KE1_, KStates_KE2_, temperature)

    diagonal_matrix = utils.varible(torch.eye(batch_size), gpu).unsqueeze(0)
    zero_vec = torch.zeros_like(inter_similarity_matrix)

    # obtain positive pair of KStates
    pos_KStates = torch.where(diagonal_matrix > 0, inter_similarity_matrix, zero_vec).sum(dim=-1)
    # obtain negivate pair of KStates
    neg_intra_KStates = torch.where(diagonal_matrix < 1, intra_similarity_matrix, zero_vec).sum(dim=-1)
    neg_inter_KStates = torch.where(diagonal_matrix < 1, inter_similarity_matrix, zero_vec).sum(dim=-1)

    ttl_KStates = pos_KStates + neg_intra_KStates + neg_inter_KStates
    cl_loss_KStates = -torch.log(pos_KStates / ttl_KStates)
    return torch.mean(cl_loss_KStates)

def KnowledgeLossOfSimilarityMatrix(matrix1, matrix2, temperature):
    matrix1_ = matrix1.transpose(1, 0)
    matrix2_ = matrix2.transpose(1, 0)
    similarity_matrix = torch.exp(torch.matmul(matrix1_, torch.transpose(matrix2_, 1, 2)) / temperature)
    return similarity_matrix

def lossOfRepresentationLevel(adj_exercise_kc, adj_EE_view, adj_KK_view, exercise_embedding, kc_ebedding, contrastive_exercises, contrastive_KCs, temperature, gpu):
    kc_ebedd, contrastive_kc_ebedd = F.normalize(kc_ebedding, dim=1), F.normalize(contrastive_KCs, dim=1)
    exercise_ebedd, contrastive_exercise_ebedd = F.normalize(exercise_embedding, dim=1), F.normalize(
        contrastive_exercises, dim=1)

    # CL_loss of knowledge concepts
    intra_kc = torch.exp(torch.matmul(kc_ebedd, kc_ebedd.t()) / temperature)
    inter_kc = torch.exp(torch.matmul(kc_ebedd, contrastive_kc_ebedd.t()) / temperature)
    diagonal_matrix_KC = utils.varible(torch.eye(adj_KK_view.shape[0]), gpu)
    zero_vec = torch.zeros_like(inter_kc)
    pos_kc = torch.where(diagonal_matrix_KC > 0, inter_kc, zero_vec).sum(dim=-1)
    neg_intra_kc = torch.where(diagonal_matrix_KC < 1, intra_kc, zero_vec).sum(dim=-1)
    neg_inter_kc = torch.where(diagonal_matrix_KC < 1, inter_kc, zero_vec).sum(dim=-1)
    ttl_kc = pos_kc + neg_intra_kc + neg_inter_kc
    cl_loss_kc = -torch.log(pos_kc / ttl_kc)

    # CL_loss of exercises
    intra_exercise = torch.exp(torch.matmul(exercise_ebedd, exercise_ebedd.t()) / temperature)
    inter_exercise = torch.exp(torch.matmul(exercise_ebedd, contrastive_exercise_ebedd.t()) / temperature)
    diagonal_matrix_exercise = utils.varible(torch.eye(adj_EE_view.shape[0]), gpu)
    zero_vec = torch.zeros_like(inter_exercise)
    pos_exercise = torch.where(diagonal_matrix_exercise > 0, inter_exercise, zero_vec).sum(dim=-1)
    neg_intra_exercise = torch.where(diagonal_matrix_exercise < 1, intra_exercise, zero_vec).sum(dim=-1)
    neg_inter_exercise = torch.where(diagonal_matrix_exercise < 1, inter_exercise, zero_vec).sum(dim=-1)

    ttl_exercise = pos_exercise + neg_intra_exercise + neg_inter_exercise
    cl_loss_exercise = -torch.log(pos_exercise / ttl_exercise)

    cl_loss = torch.mean(cl_loss_exercise) + torch.mean(cl_loss_kc)
    return cl_loss