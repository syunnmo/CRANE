import numpy as np
import math
import torch
from time import time
from torch import nn
import utils as utils
from sklearn import metrics
import csv
from tqdm import tqdm

def train(epoch_num, model, params, optimizer, scheduler, adj_exercise_kc, adj_EE_view, adj_KK_view, train_kc_data, train_exercise_data, train_exercise_respond_data):
    N = int(math.floor(len(train_exercise_data) / params.batch_size))
    shuffle_index = np.random.permutation(train_exercise_data.shape[0])
    train_kc_data = train_kc_data[shuffle_index]
    train_exercise_data = train_exercise_data[shuffle_index]
    train_exercise_respond_data = train_exercise_respond_data[shuffle_index]

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()
    adj_exercise_kc = utils.varible(torch.from_numpy(adj_exercise_kc), params.gpu)
    adj_EE_view = utils.varible(torch.from_numpy(adj_EE_view), params.gpu)
    adj_KK_view = utils.varible(torch.from_numpy(adj_KK_view), params.gpu)

    for idx in tqdm (range(N), desc='Training'):
        data_length = train_kc_data.shape[0]
        begin, end = idx * params.batch_size, min((idx + 1) * params.batch_size, data_length - 1)

        kc_one_seq = train_kc_data[begin : end, :]
        exercise_one_seq = train_exercise_data[begin : end, :]
        exercise_respond_batch_seq = train_exercise_respond_data[begin : end, :]
        target = train_exercise_respond_data[begin : end, :]

        target = (target - 1) / params.n_exercise
        target = np.floor(target)

        input_kc = utils.varible(torch.LongTensor(kc_one_seq), params.gpu)
        input_exercise = utils.varible(torch.LongTensor(exercise_one_seq), params.gpu)
        input_exercise_respond = utils.varible(torch.LongTensor(exercise_respond_batch_seq), params.gpu)
        target = utils.varible(torch.FloatTensor(target), params.gpu)
        # Remove the first question
        target = target[ : , 1 : ]
        target_to_1d = torch.chunk(target, target.shape[0], 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(target.shape[0])], 1)
        target_1d = target_1d.permute(1, 0)

        model.zero_grad()
        loss, filtered_pred, filtered_target, exercise_embedding, kc_embedding = model.forward(adj_exercise_kc, adj_EE_view, adj_KK_view, input_kc, input_exercise, input_exercise_respond, target_1d)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), params.maxgradnorm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())

        pred_list.append(right_pred)
        target_list.append(right_target)

    # Save exercise embedding
    # times = time()
    # torch.save(exercise_embedding, "./data/assist2009_B/exercise_embedding" + str(times) + ".pth")

    # Save KC embedding
    times = time()
    torch.save(kc_embedding, "./data/assist2009_B/KC_embedding" + str(times) + ".pth")

    scheduler.step()
    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    MSE = metrics.mean_squared_error(all_target, all_pred)
    RMSE = math.sqrt(MSE)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc, RMSE


def test(model, params, optimizer, scheduler, adj_exercise_kc, adj_EE_view, adj_KK_view, kc_data, exercise_data, exercise_respond_data):
    N = int(math.floor(len(exercise_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()
    adj_exercise_kc = utils.varible(torch.from_numpy(adj_exercise_kc), params.gpu)
    adj_EE_view = utils.varible(torch.from_numpy(adj_EE_view), params.gpu)
    adj_KK_view = utils.varible(torch.from_numpy(adj_KK_view), params.gpu)

    for idx in tqdm(range(N), desc='Validing/testing'):
        data_length = kc_data.shape[0]
        begin, end = idx * params.batch_size, min((idx + 1) * params.batch_size, data_length - 1)

        kc_one_seq = kc_data[begin : end, :]
        exercise_one_seq = exercise_data[begin : end, :]
        exercise_respond_batch_seq = exercise_respond_data[begin : end, :]
        target = exercise_respond_data[begin : end, :]

        target = (target - 1) / params.n_exercise
        target = np.floor(target)

        input_kc = utils.varible(torch.LongTensor(kc_one_seq), params.gpu)
        input_exercise = utils.varible(torch.LongTensor(exercise_one_seq), params.gpu)
        input_exercise_respond = utils.varible(torch.LongTensor(exercise_respond_batch_seq), params.gpu)
        target = utils.varible(torch.FloatTensor(target), params.gpu)
        # Remove the first question
        target = target[:, 1:]
        target_to_1d = torch.chunk(target, target.shape[0], 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(target.shape[0])], 1)
        target_1d = target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target, exercise_embedding, kc_embedding = model.forward(adj_exercise_kc, adj_EE_view, adj_KK_view, input_kc, input_exercise, input_exercise_respond, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    with open('P_statistics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(all_pred)  # 第一行
        writer.writerow(all_target)

    auc = metrics.roc_auc_score(all_target, all_pred)
    MSE = metrics.mean_squared_error(all_target, all_pred)
    RMSE = math.sqrt(MSE)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)
    # f1 = metrics.f1_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc, RMSE









