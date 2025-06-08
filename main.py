import torch
import argparse
from model import MODEL
from run import train, test
import numpy as np
import math
from torch.optim import Adam, lr_scheduler
from dataloader import getDataLoader

def generate_KCs_view(data_dir, adj_exercise_kc):
    n_exercise = adj_exercise_kc.shape[0]
    n_kc = adj_exercise_kc.shape[1]
    adj = np.eye(n_kc, k=0, dtype=np.int)
    for i in range(n_exercise):
        indexList = []
        for j in range(n_kc):
            if adj_exercise_kc[i][j] == 1:
                indexList.append(j)
        for k in range(len(indexList)):
            for l in range(len(indexList)):
                adj [indexList[k]][indexList[l]] = 1
    np.savetxt(data_dir + "/adj_KK_view.txt", adj, fmt="%d")

def generate_exercises_view(data_dir, adj_exercise_kc):
    n_exercise = adj_exercise_kc.shape[0]
    n_kc = adj_exercise_kc.shape[1]
    adj = np.eye(n_exercise, k=0, dtype=np.int)
    for i in range(n_kc):
        indexList = []
        for j in range(n_exercise):
            if adj_exercise_kc[j][i] == 1:
                indexList.append(j)
        for k in range(len(indexList)):
            for l in range(len(indexList)):
                adj [indexList[k]][indexList[l]] = 1
    np.savetxt(data_dir + "/adj_EE_view.txt", adj, fmt="%d")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='the gpu will be used, e.g "0,1,2"')
    parser.add_argument('--max_iter', type=int, default=500, help='number of iterations')
    parser.add_argument('--init_lr', type=float, default= 0.001, help='initial learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='The parameter of LeakyReLU')
    parser.add_argument('--exercise_embed_dim', type=int, default=128, help='exercise embedding dimensions')
    parser.add_argument('--hidden_dim', type=float, default=256, help='knowledge state dim')
    parser.add_argument('--layer_num', type=float, default=1, help='layer number for LSTM')
    parser.add_argument('--max_step', type=int, default=200, help='the allowed maximum length of a sequence')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.8,help='The temperature of Contrastive Learning')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of head attentions. Such as 1,2,4,6,8')
    parser.add_argument('--interval', type=int, default=10, help='Interval of updating the learning efficiency')
    parser.add_argument('--coldstart', type=float, default=1.0, help='Training set for training')
    parser.add_argument('--lambda1', type=float, default= 0.5, help='Weight parameters of the loss function')
    parser.add_argument('--fold', type=str, default='1', help='number of fold')
    dataset = 'assist2009_B'

    if dataset == 'assist2009_B':
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--n_knowledge_concept', type=int, default=110, help='the number of unique questions in the dataset')
        parser.add_argument('--n_exercise', type=int, default=16891, help='the number of unique questions in the dataset')
        parser.add_argument('--data_dir', type=str, default='./data/assist2009_B', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2009_B', help='data set name')

    if dataset == 'assist2017':
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--n_knowledge_concept', type=int, default=102, help='the number of unique questions in the dataset')
        parser.add_argument('--n_exercise', type=int, default=3162, help='the number of unique questions in the dataset')
        parser.add_argument('--data_dir', type=str, default='./data/assist2017', help='data directory')
        parser.add_argument('--data_name', type=str, default='assist2017', help='data set name')

    if dataset == 'ednet':
        parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
        parser.add_argument('--n_knowledge_concept', type=int, default=188, help='the number of unique questions in the dataset')
        parser.add_argument('--n_exercise', type=int, default=11427, help='the number of unique questions in the dataset')
        parser.add_argument('--data_dir', type=str, default='./data/ednet', help='data directory')
        parser.add_argument('--data_name', type=str, default='ednet', help='data set name')

    params = parser.parse_args()
    params.memory_size = params.n_knowledge_concept
    params.lr = params.init_lr
    params.memory_key_state_dim = params.exercise_embed_dim
    params.memory_value_state_dim = params.exercise_embed_dim * 2

    train_data_path = params.data_dir + "/" + params.data_name + "_train"+ params.fold + ".csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid"+ params.fold + ".csv"
    test_data_path = params. data_dir + "/" + params.data_name + "_test"+ params.fold + ".csv"

    train_kc_data, train_respond_data, train_exercise_data, \
    valid_kc_data, valid_respose_data, valid_exercise_data, \
    test_kc_data, test_respose_data, test_exercise_data \
        = getDataLoader(train_data_path, valid_data_path, test_data_path, params)

    # for cold start
    sumLearners = train_kc_data.shape[0]
    usedTranin = math.ceil(sumLearners * params.coldstart)
    train_kc_data = train_kc_data[0:usedTranin, :]
    train_respond_data = train_respond_data[0:usedTranin, :]
    train_exercise_data = train_exercise_data[0:usedTranin, :]
    if params.coldstart < 1.0 :
        print('using ' + str(params.coldstart) + ' for cold start training')


    train_exercise_respond_data = train_respond_data * params.n_exercise + train_exercise_data
    valid_exercise_respose_data = valid_respose_data * params.n_exercise + valid_exercise_data
    test_exercise_respose_data = test_respose_data * params.n_exercise + test_exercise_data


    # obtain KS
    adj_exercise_kc = np.loadtxt(params.data_dir + "/exercise_kc_map.txt")

    # # You should release the following comment if views are not generated
    # # exercise-exercise view and KC-KC view generated
    # adj_EE_view = generate_exercises_view(params.data_dir, adj_exercise_kc)
    # adj_KK_view = generate_KCs_view(params.data_dir, adj_exercise_kc)

    try:
        # load views
        adj_EE_view = np.loadtxt(params.data_dir + "/adj_EE_view.txt", dtype=np.float32)
        adj_KK_view = np.loadtxt(params.data_dir + "/adj_KK_view.txt", dtype=np.float32)
    except:
        print("Views are not generated")

    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)

    model = MODEL(n_exercise=params.n_exercise,
                  batch_size=params.batch_size,
                  exercise_embed_dim=params.exercise_embed_dim,
                  hidden_dim = params.hidden_dim,
                  layer_num = params.layer_num,
                  interval = params.interval,
                  params = params )

    optimizer = Adam(params=model.parameters(), lr=params.lr, betas=(0.9, 0.9), weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model.cuda()

    best_test_auc = 0
    best_test_acc= 0
    best_test_RMSE = 1.0

    for idx in range(params.max_iter):
        train_loss, train_accuracy, train_auc, train_RMSE = train(idx, model, params, optimizer, scheduler, adj_exercise_kc, adj_EE_view, adj_KK_view, train_kc_data, train_exercise_data, train_exercise_respond_data)
        print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f, RMSE : %3.5f' % (idx + 1, params.max_iter, train_loss, train_auc, train_accuracy, train_RMSE))
        with torch.no_grad():
            valid_loss, valid_accuracy, valid_auc, valid_RMSE = test(model, params, optimizer, scheduler, adj_exercise_kc, adj_EE_view, adj_KK_view, valid_kc_data, valid_exercise_data, valid_exercise_respose_data)
            print('Epoch %d/%d, loss : %3.5f, valid auc : %3.5f, valid accuracy : %3.5f, valid RMSE : %3.5f' % (idx + 1, params.max_iter, valid_loss, valid_auc, valid_accuracy, valid_RMSE))
            test_loss, test_accuracy, test_auc, test_RMSE = test(model, params, optimizer, scheduler, adj_exercise_kc, adj_EE_view, adj_KK_view, test_kc_data, test_exercise_data, test_exercise_respose_data)
            print("test_loss: %.4f\t test_auc: %.4f\t test_accuracy: %.4f\t test_RMSE: %.4f\t " % (test_loss, test_auc, test_accuracy, test_RMSE))

        if test_auc > best_test_auc:
            print('auc: %3.4f to %3.4f' % (best_test_auc, test_auc))
            best_test_auc = test_auc

        if test_accuracy > best_test_acc:
            print('acc: %3.4f to %3.4f' % (best_test_acc, test_accuracy))
            best_test_acc = test_accuracy

        if test_RMSE < best_test_RMSE:
            print('RMSE: %3.4f to %3.4f' % (best_test_RMSE, test_RMSE))
            best_test_RMSE = test_RMSE

    print("temperature: %.2f\t lamda: %.2f\t  test_auc: %.4f\t test_accuracy: %.4f\t test_RMSE: %.4f\t" % (params.temperature, params.Lamda, best_test_auc, best_test_acc, best_test_RMSE))

if __name__ == "__main__":
    main()
