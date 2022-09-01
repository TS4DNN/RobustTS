import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import numpy as np
from scipy.special import softmax
import torch
from torch_cluster import knn
import torch.optim as optim
from scipy.spatial import distance
import time
import copy
import random
from tqdm import tqdm
from typing import Optional
from torch_geometric.data import Data, DataLoader
from utils import AverageMeter, RecorderMeter
import os
import csv
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.manifold import TSNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


default_hyperparams = {
    'custom': {'img_size': 32, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 600,
               'retrain_epoch': 10, 'feature_dim': 512},
    'svhn': {'img_size': 32, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 600,
             'retrain_epoch': 10, 'feature_dim': 512},
    'cifar10': {'img_size': 32, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 600,
                'retrain_epoch': 10, 'feature_dim': 512},
    'stl10': {'img_size': 96, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 600,
              'retrain_epoch': 10, 'feature_dim': 512},

}


def test_rank_prior(x, y, x_candidate, model, byol_model, save_path, args):
    latents = byol_model(x)
    st = time.time()
    no_neighbors = 100
    labeled_indices = [0]
    unlabeled_indices = [1]
    neg_case_indexes = []
    pos_case_indexes = []
    p_budget_lst = [1, 2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # percentage of budget
    selected = set()
    misclass_array = []
    correct_array = []
    logits = []
    budget_lst = []
    pfd_lst = []
    ideal_pfd_lst = []
    budget_lst.insert(0, 0)
    pfd_lst.insert(0, 0)
    ideal_pfd_lst.insert(0, 0)
    # Iterative
    print("# of unlabeled test inputs: {}".format(len(unlabeled_indices)))

    # get classification result rank
    prob = softmax(logits, axis=1)
    confidence = np.sum(np.square(prob), axis=1)
    prob = torch.from_numpy(prob)

    # method2: apply GNN classification algorithm
    # old_time = time.time()
    hidden_dim = default_hyperparams['cifar10']['gnn_hidden_dim']
    epochs = default_hyperparams['cifar10']['gnn_train_epoch']
    # create dataset
    print('Start runing GNN classification algorithm')

    # construct knn graph
    # original version
    # batch = torch.tensor([0 for _ in range(latents.shape[0])])
    # edge_index = knn_graph(torch.from_numpy(latents).float().to(device),
    #                         batch=batch.to(device),
    #                         k=no_neighbors,
    #                         cosine=True, loop=False)

    # print("edge_index: ", edge_index[:10])
    # new_time = time.time()
    # print("Finish calculate edge index, the shape is {}, time cost: {}".format(edge_index.shape, new_time-old_time), log)

    # approximate version
    x_l_indices = labeled_indices
    x_u_indices = unlabeled_indices
    x_l = torch.from_numpy(latents[x_l_indices]).float().to(device)
    x_u = torch.from_numpy(latents[x_u_indices]).float().to(device)

    batch = torch.tensor([0 for _ in range(x_l.shape[0])]).to(device)
    edge_index_t = my_knn_graph(x_l, x_l, batch_x=batch, batch_y=batch, cosine=True, loop=False, k=no_neighbors)
    print("l-2-l edge index: ", edge_index_t)
    # edge_index = torch.zeros_like(edge_index_t)
    new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
    new_edge_index_l1 = [x_l_indices[i] for i in list(edge_index_t[1])]
    l2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
    print("replaced edge index: ", l2l_edge_index)

    batch_x = torch.tensor([0 for _ in range(x_l.shape[0])]).to(device)
    batch_y = torch.tensor([0 for _ in range(x_u.shape[0])]).to(device)
    edge_index_t = my_knn_graph(x_l, x_u, batch_x=batch_x, batch_y=batch_y, cosine=True, loop=False, k=no_neighbors)
    print("u-2-l edge index: ", edge_index_t)
    new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
    new_edge_index_l1 = [x_u_indices[i] for i in list(edge_index_t[1])]
    u2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
    print("replaced edge index: ", u2l_edge_index)

    edge_index = torch.cat([l2l_edge_index, u2l_edge_index], dim=1)
    print("final edge_index, ", u2l_edge_index)
    print(
        "Finish calculate edge index, the shape is {}, time cost: {:4f}".format(edge_index.shape, time.time() - st))
    # end approximation

    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    for i in range(edge_index.size(1)):
        edge_weight[i] = distance.cosine(latents[edge_index[0, i]], latents[edge_index[1, i]])
    print('Finish calculating edge weight. example of edge_weight: {}'.format(edge_weight[:10]))
    # print("time cost to calculate edge weight: {}".format(time.time()-new_time), log)
    # return

    y = torch.zeros((latents.shape[0]), dtype=torch.long)
    imbalance_ratio = 1. * len(neg_case_indexes) / len(pos_case_indexes)
    print("neg:pos imbalance ratio: {}".format(imbalance_ratio))

    # under-sampling
    # neg_case_indexes = random.sample(neg_case_indexes, len(pos_case_indexes))
    # class_weights = torch.tensor([1., 1.])

    # importance sampling
    class_weights = torch.tensor([1. / (1 + imbalance_ratio), 1. * imbalance_ratio / (1 + imbalance_ratio)])
    # print("class_weights: {}".format(class_weights), log)

    y[pos_case_indexes] = 1
    y[neg_case_indexes] = 0
    print('positives: {}'.format(len(pos_case_indexes)))
    print('negatives: {}'.format(len(neg_case_indexes)))

    dataset = Data(x=torch.from_numpy(latents).float(), y=y, edge_index=edge_index)
    dataset.edge_weight = edge_weight
    print('example edge index: {}'.format(dataset.edge_index))
    print('example edge index: max {} min {}'.format(dataset.edge_index[0].max(), dataset.edge_index[0].min()))
    # print ('example x: {}'.format(dataset.x[:10]))
    print('example y: {}'.format(dataset.y.sum()))

    print("dataset info: {}".format(dataset))
    dataset.num_classes = y.max().item() + 1
    print("**number of classes: {}".format(dataset.num_classes))

    # split train/val/test data
    dataset.train_mask = torch.zeros((latents.shape[0],), dtype=torch.bool)
    dataset.val_mask = torch.zeros((latents.shape[0],), dtype=torch.bool)
    dataset.test_mask = torch.zeros((latents.shape[0],), dtype=torch.bool)

    labeled_list = list(neg_case_indexes + pos_case_indexes)
    random.shuffle(labeled_list)
    dataset.train_mask[labeled_list[:int(0.8 * len(labeled_list))]] = True
    dataset.val_mask[labeled_list[int(0.8 * len(labeled_list)):]] = True
    dataset.test_mask[list(set(range(latents.shape[0])) - set(labeled_list))] = True

    # logging
    print("GNN training info")
    print("number of train cases: {}".format(dataset.train_mask.sum().item()))
    print("number of val cases: {}".format(dataset.val_mask.sum().item()))
    print("number of test cases: {}".format(dataset.test_mask.sum().item()))
    print("GNN input dimension: {}".format(dataset.num_node_features))
    print("GNN output dimension: {}".format(dataset.num_classes))

    # build gnn model
    gcn_model = GNNStack(input_dim=max(dataset.num_node_features, 1),
                         hidden_dim=hidden_dim,
                         output_dim=dataset.num_classes)
    opt = optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=5e-4)

    # build mlp model
    # if args.learn_mixed:
    print('shape of sample embedding: ', hidden_dim + prob.shape[1] + 1)
    mlp_model = MLP(hidden_dim + prob.shape[1] + 1, 2).to(device)
    lc_optimizer = optim.SGD(mlp_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    mlp_best_loss = 100
    mlp_model_path = "mlp_model.ckpt"

    recorder = RecorderMeter(epochs)
    mlp_recorder = RecorderMeter(epochs)
    dynamic_dataset = copy.deepcopy(dataset)
    for e_iter in range(epochs):
        mlp_best_loss = train_graph(gcn_model, mlp_model,
                                    confidence, prob, lc_optimizer,
                                    mlp_best_loss, mlp_model_path, criterion,
                                    opt=opt,
                                    dataset=dataset, dynamic_dataset=dynamic_dataset,
                                    class_weights=class_weights,
                                    recorder=recorder, mlp_recorder=mlp_recorder,
                                    epoch=e_iter, log="log", args=args)
    recorder.plot_curve(os.path.join(save_path, 'gcn_train_curve.pdf'))
    mlp_recorder.plot_curve(os.path.join(save_path, 'mlp_train_curve.pdf'))

    # test
    # with torch.no_grad():
    #     emb, output_distribution = gcn_model_3(dynamic_dataset.x, dynamic_dataset.edge_index, edge_weight=dynamic_dataset.edge_weight)
    #     output_distribution  = torch.exp(output_distribution)
    #     # print("output distribution: {}".format(output_distribution.shape), log=log)
    # output_distribution = output_distribution.cpu().numpy()
    gcn_model.eval()
    with torch.no_grad():
        emb, output_distribution = gcn_model(dataset.x, dataset.edge_index, edge_weight=dataset.edge_weight)
        output_distribution = torch.exp(output_distribution)
        # print("output distribution: {}".format(output_distribution.shape), log=log)
    output_distribution = output_distribution.cpu().numpy()

    # if args.bf_mixed:
    #     # collaborative judgement of confidence and GNN
    #     print("Mixed method enabled: combine gini and GNN based method")
    #     mix_rank_indicator = output_distribution[:, 1] * (
    #                 1 - confidence)  # the bigger, the more likely to be positive case
    #     ranked_indexes = np.argsort(mix_rank_indicator)[::-1].astype(
    #         np.int64)  # cases having high positive probability are put in the front
    # elif args.learn_mixed:
    print("Mixed method enabled: combine gini and correlation based method, learn based")
    print("dim: {}/{}/{}".format(emb.shape, torch.tensor(confidence).unsqueeze(1).shape, prob.shape))
    # visualize the learned embedding by gcn_model
    # codes_embedded = TSNE(n_components=2).fit_transform(emb.detach().cpu()[class_inds])
    # plot_2d_scatter(codes_embedded, misclass_array[class_inds], save_path=save_path,
    #                 fig_name='gcn_embedding')

    sample_emd = torch.cat([emb.to(device), torch.tensor(confidence).unsqueeze(1).to(device), prob.to(device)],
                           dim=1).type(torch.FloatTensor)

    # # visualize the learned embedding by gcn_model
    # if args.latent_space_plot:
    #     codes_embedded = TSNE(n_components=2).fit_transform(sample_emd.detach().cpu()[class_inds])
    #     plot_2d_scatter(codes_embedded, misclass_array[class_inds], save_path=save_path,
    #                     fig_name='mix_embedding')

        # test
    # load best model from checkpoint
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
    outputs = mlp_model(sample_emd.to(device))
    output_distribution = F.softmax(outputs.cpu(), dim=1)
    # print (output_distribution)

    output_distribution = output_distribution.detach().numpy()
    ranked_indexes = np.argsort(output_distribution[:, 1])[::-1].astype(np.int64)
    # else:
    ranked_indexes = np.argsort(output_distribution[:, 1])[::-1].astype(np.int64)

    index2select = [i for i in ranked_indexes if ((i not in selected) and (i not in labeled_indices))]
    print("ranked output distri for selection: {}".format(output_distribution[:, 1][index2select][:100]))

    for p_budget in p_budget_lst:
        print("\n ###### budget percent is {}% ######".format(p_budget))
        selected_temp = copy.deepcopy(selected)
        # model2test_temp = copy.deepcopy(model2test)
        neg_case_indexes_temp = copy.deepcopy(neg_case_indexes)
        pos_case_indexes_temp = copy.deepcopy(pos_case_indexes)

        budget = int(p_budget * len(unlabeled_indices) / 100.0)

        available_slots = budget - len(selected_temp)
        sel_indexes = index2select[:available_slots]
        sel_indexes = np.array(sel_indexes)

        # Step 3: test
        selected_temp.update(set(sel_indexes))
        pos_count = misclass_array[sel_indexes].sum()
        neg_t = list(sel_indexes[np.nonzero(correct_array[sel_indexes])[0]])
        pos_t = list(sel_indexes[np.nonzero(misclass_array[sel_indexes])[0]])

        neg_case_indexes_temp += neg_t
        pos_case_indexes_temp += pos_t

        assert (len(neg_case_indexes_temp) == len(
            set(neg_case_indexes_temp)))  # make sure no duplicated elements in the selected list
        assert (len(pos_case_indexes_temp) == len(set(pos_case_indexes_temp)))

        # logging
        print('==> total selected count.: {}'.format(len(selected_temp)))
        print('==> total pos count: {}'.format(pos_count))
        # print('==> current selected indexes: {}'.format(sel_indexes), log )
        print('    -- positive count: {}'.format(misclass_array[sel_indexes].sum()))
        print('    -- neg_case_indexes length: {}'.format(len(neg_t)))
        print('    -- pos_case_indexes length: {}'.format(len(pos_t)))

        # evaluation metric 1
        print("pos_count: {}".format(pos_count))
        print("total bug count: {}".format(misclass_array[unlabeled_indices].sum()))
        p_fault_detected = 100.0 * pos_count / misclass_array[unlabeled_indices].sum()
        ideal_fault_detected = 100.0 * budget / misclass_array[unlabeled_indices].sum()
        if ideal_fault_detected > 100.0:
            ideal_fault_detected = 100.00
        random_p_fault_detected = (100.0 * budget / misclass_array[unlabeled_indices].shape[0])

        print("Model2test: {}".format(args.model2test_path))
        print("Model2Test Accuracy on labeled data: {}".format(
            100.0 * correct_array[labeled_indices].sum() / misclass_array[labeled_indices].shape[0]))
        print("Total faults: {}".format(misclass_array[unlabeled_indices].sum()))
        print("Total test cases: {}".format(len(unlabeled_indices)))
        print("Percentage of fault detected: %s " % (p_fault_detected))
        print("Percentage of fault detected (random): %s " % (random_p_fault_detected))

        # output and logging
        out_file = os.path.join(save_path, 'gnn_result.csv' if args.graph_nn else 'lp_result.csv')
        print("writing output to csv file: {}".format(out_file))

        budget_lst.append(p_budget / 100.0)
        pfd_lst.append(p_fault_detected / 100.0)
        ideal_pfd_lst.append(ideal_fault_detected / 100.0)

        apfd = 0
        ideal_apfd = 0
        if p_budget == 100:
            apfd = get_APFD(copy.copy(budget_lst), pfd_lst)
            ideal_apfd = get_APFD(budget_lst, ideal_pfd_lst)

        with open(out_file, 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                args.model2test_arch,
                args.model2test_path,
                budget,
                p_budget,
                args.sel_method,
                no_neighbors,
                'FaultDetected',
                p_fault_detected,
                ideal_fault_detected,
                random_p_fault_detected,
                'APFD',
                apfd,
                ideal_apfd,
                'TRC',
                p_fault_detected / ideal_fault_detected
            ])

        # overlap with gini
        conf_ranked_indexes = np.argsort(confidence)
        conf_index2select = [i for i in conf_ranked_indexes if i not in labeled_indices]
        confidence_selected = conf_index2select[:budget]

        overlap_ratio = 1.0 * len(set(confidence_selected).intersection(set(selected_temp))) / len(confidence_selected)
        print("overlap ratio: " + str(overlap_ratio))

    print('success!')

    return


class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.2
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return pyg_nn.GCNConv(input_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight):
        x, edge_index = x, edge_index

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            emb = x
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)
        x = self.post_mp(x)

        return emb, F.log_softmax(x, dim=1)

    def loss(self, pred, label, weight):
        return F.nll_loss(input=pred, target=label, weight=weight)


def my_knn_graph(x: torch.Tensor, y:torch.Tensor, k: int,
                batch_x: Optional[torch.Tensor] = None,
                batch_y: Optional[torch.Tensor] = None,
                loop: bool = False, flow: str = 'source_to_target',
                cosine: bool = False, num_workers: int = 1) -> torch.Tensor:

    assert flow in ['source_to_target', 'target_to_source']
    # Finds for each element in :obj:`y` the :obj:`k` nearest points in obj:`x`.
    edge_index = knn(x, y, k if loop else k + 1, batch_x, batch_y, cosine,
                     num_workers)

    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


class MLP(nn.Module):
    def __init__(self, num_ftrs, out_dim):
         super(MLP, self).__init__()
         self.l1 = nn.Linear(num_ftrs, num_ftrs)
         self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x


def train_graph(gcn_model, mlp_model, confidence, prob, lc_optimizer, mlp_best_loss, mlp_model_path, criterion,
                opt, dataset, dynamic_dataset, class_weights, recorder, mlp_recorder, log, epoch, args):
    global no_neighbors

    gcn_model = gcn_model.to(device)
    dataset.x, dataset.edge_index = dataset.x.to(device), dataset.edge_index.to(device)
    dataset.y = dataset.y.to(device)
    dataset.edge_weight = dataset.edge_weight.to(device)
    dataset.train_mask = dataset.train_mask.to(device)
    dataset.val_mask = dataset.val_mask.to(device)
    dataset.test_mask = dataset.test_mask.to(device)
    batch = torch.tensor([0 for _ in range(dataset.x.shape[0])]).to(device)

    # train
    gcn_model.train()
    t = time.time()
    gcn_correct = 0
    total = 0
    gcn_train_loss = 0
    opt.zero_grad()

    mlp_start_epoch = 450

    # gcn 1 loss
    # if epoch < 150:
    emb, pred = gcn_model(dataset.x, dataset.edge_index, edge_weight=dataset.edge_weight)

    label = dataset.y[dataset.train_mask]
    pred = pred[dataset.train_mask]

    gcn_loss = gcn_model.loss(pred, label, class_weights.to(device))
    # gcn train acc
    gcn_train_loss += gcn_loss.item()
    gcn_correct += pred.argmax(1).eq(label).sum().item()
    total += len(label)

    gcn_train_acc = 100.0 * gcn_correct / total

    # mlp loss
    if args.learn_mixed and epoch >= mlp_start_epoch:

        mlp_train_correct = 0
        mlp_train_loss = 0

        sample_emd = torch.cat([emb.detach(), torch.tensor(confidence).unsqueeze(1).to(device),
                                prob.to(device)], dim=1).type(torch.FloatTensor)
        train_set = Dataset(sample_emd, dataset.y, dataset.train_mask)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=128)
        for local_data, local_labels in train_loader:
            lc_optimizer.zero_grad()
            outputs = mlp_model(local_data.to(device))
            mlp_loss = criterion(outputs, local_labels.to(device))
            mlp_train_loss += mlp_loss.cpu().item()
            _, predicted = outputs.max(axis=1)
            mlp_train_correct += predicted.eq(local_labels).sum().item()
            # train gcn and mlp together
            # loss = gcn_loss + mlp_loss
            # loss.backward(retain_graph=True)
            # lc_optimizer.step()
            # outputs = None
            # train mlp alone
            loss = mlp_loss
            loss.backward()
            lc_optimizer.step()
        mlp_train_acc = 100.0 * mlp_train_correct / total
        # train gcn and mlp together
        # opt.step()
    else:
        loss = gcn_loss
        loss.backward()
        opt.step()

    # val gcn
    gcn_val_acc, gcn_precision, gcn_recall, gcn_f1_score = test_graph(dataset, gcn_model, is_validation=True)

    if epoch < mlp_start_epoch:
        print('GCN MODEL Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(gcn_train_loss),
              'acc_train: {:.4f}'.format(gcn_train_acc),
              'acc_val: {:.4f}'.format(gcn_val_acc),
              'precision: {:.4f}'.format(gcn_precision),
              'recall: {:.4f}'.format(gcn_recall),
              'f1_score: {:.4f}'.format(gcn_f1_score),
              'time: {:.4f}s'.format(time.time() - t))
        recorder.update(epoch,
                        train_loss=gcn_train_loss,
                        train_acc=gcn_train_acc,
                        val_loss=0,
                        val_acc=gcn_val_acc)

    # val mlp
    if args.learn_mixed and epoch >= mlp_start_epoch:
        mlp_model.eval()
        mlp_val_loss = 0
        # total = 0
        # mlp_val_correct = 0
        sample_emd = torch.cat(
            [emb[dataset.val_mask], torch.tensor(confidence).unsqueeze(1)[dataset.val_mask].to(device),
             prob[dataset.val_mask].to(device)], dim=1).type(torch.FloatTensor)
        local_batch, local_labels = sample_emd, dataset.y[dataset.val_mask]
        outputs = mlp_model(local_batch.to(device))
        loss = criterion(outputs, local_labels)
        mlp_val_loss += loss.cpu().item()

        _, predicted = outputs.max(1)
        mlp_val_correct = predicted.eq(local_labels).sum().item()
        total = local_labels.size(0)
        mlp_val_acc = 100.0 * mlp_val_correct / total

        if mlp_val_loss < mlp_best_loss:
            mlp_best_loss = mlp_val_loss
            # save checkpoint
            print('saving checkpoint...')
            torch.save(mlp_model.state_dict(), mlp_model_path)

        print('MLP MODEL Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(mlp_train_loss),
              'acc_train: {:.4f}'.format(mlp_train_acc),
              'acc_val: {:.4f}'.format(mlp_val_acc),
              'time: {:.4f}s'.format(time.time() - t))

        mlp_recorder.update(int((epoch - mlp_start_epoch)),
                            train_loss=mlp_train_loss,
                            train_acc=mlp_train_acc,
                            val_loss=mlp_val_loss,
                            val_acc=mlp_val_acc)

    return mlp_best_loss


def test_graph(dataset, model, is_validation=False):
    global no_neighbors
    model.eval()
    dataset = dataset.to(device)
    correct = 0
    with torch.no_grad():
        emb, pred = model(dataset.x, dataset.edge_index, dataset.edge_weight)
        pred = pred.argmax(dim=1)

        mask = dataset.val_mask if is_validation else dataset.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = dataset.y[mask]
        # print ("val label ratio: ", np.array(label.cpu()==0).sum()/np.array(label.cpu()==1).sum())

        correct += pred.eq(label).sum().item()

    total = len(label)

    precision = sklearn.metrics.precision_score(y_true=label.cpu(), y_pred=pred.cpu(), zero_division='warn')
    recall = sklearn.metrics.recall_score(y_true=label.cpu(), y_pred=pred.cpu())
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    if is_validation == False:
        print("consufion matrix:\n ", sklearn.metrics.confusion_matrix(y_true=label.cpu(), y_pred=pred.cpu()))
    return 100. * correct / total, precision, recall, f1_score


def get_APFD(budget_lst, pfd_lst):
    # budget_list: value 0-1
    # pfd_list: value 0-1
    # assert (budget_lst[0] == 0.0 & pfd_lst[0] == 0.0 & budget_lst[-1] == 1.0 & pfd_lst[-1] == 1.0)
    # print ('budget list, pfd_lst: ', budget_lst, pfd_lst)
    apfd = 0
    for i in range(len(budget_lst)):
        if i == 0:
            continue
        else:
            area_temp = (budget_lst[i] - budget_lst[i - 1]) * (pfd_lst[i] - pfd_lst[i - 1]) / 2 \
                        + (budget_lst[i] - budget_lst[i - 1]) * pfd_lst[i - 1]

        apfd += area_temp
    return apfd


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, data, labels, data_mask):
        'Initialization'
        self.masked_data = data[data_mask]
        self.masked_label = labels[data_mask]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.masked_label)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.masked_data[index]
        y = self.masked_label[index]

        return x, y
