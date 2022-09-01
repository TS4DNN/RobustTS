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
from tensorflow.keras.models import Model
import torchvision.models as models
import random
from typing import Optional
from torch_geometric.data import Data, DataLoader
from .utils import AverageMeter, RecorderMeter
from torch.utils.data import TensorDataset, DataLoader
import sklearn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

default_hyperparams = {
    'custom': {'img_size': 32, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 10,
               'retrain_epoch': 10, 'feature_dim': 784},
    'svhn': {'img_size': 32, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 600,
             'retrain_epoch': 10, 'feature_dim': 512},
    'cifar10': {'img_size': 32, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 600,
                'retrain_epoch': 10, 'feature_dim': 512},
    'stl10': {'img_size': 96, 'num_classes': 10, 'channel': 3, 'gnn_hidden_dim': 32, 'gnn_train_epoch': 600,
              'retrain_epoch': 10, 'feature_dim': 512},

}


def model_eval(model, x, y):
    predicted_label = model.predict(x).argmax(axis=1)
    correct_index = np.where(predicted_label == y)[0]
    correct_array = np.array([0 for i in range(len(x))])
    correct_array[correct_index] = 1
    dense_layer_model = Model(inputs=model.input, outputs=model.layers[-1].output)

    # logit_layer = K.function(inputs=model.input, outputs=model.layers[-1].output)
    logits = dense_layer_model.predict(x)
    return correct_array, logits


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
    if epoch >= mlp_start_epoch:

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
    # gcn_val_acc, gcn_precision, gcn_recall, gcn_f1_score = test_graph(dataset, gcn_model, is_validation=True)
    #
    # if epoch < mlp_start_epoch:
    #     print('GCN MODEL Epoch: {:04d}'.format(epoch + 1),
    #           'loss_train: {:.4f}'.format(gcn_train_loss),
    #           'acc_train: {:.4f}'.format(gcn_train_acc),
    #           'acc_val: {:.4f}'.format(gcn_val_acc),
    #           'precision: {:.4f}'.format(gcn_precision),
    #           'recall: {:.4f}'.format(gcn_recall),
    #           'f1_score: {:.4f}'.format(gcn_f1_score),
    #           'time: {:.4f}s'.format(time.time() - t))
    #     recorder.update(epoch,
    #                     train_loss=gcn_train_loss,
    #                     train_acc=gcn_train_acc,
    #                     val_loss=0,
    #                     val_acc=gcn_val_acc)

    # val mlp
    if epoch >= mlp_start_epoch:
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


def get_features(my_x, my_y, model):
    my_x = my_x.reshape(-1, 3, 32, 32)
    tensor_x = torch.Tensor(my_x)  # transform to torch tensor
    tensor_y = torch.Tensor(my_y)

    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # testloader = DataLoader(my_dataset)  # create your dataloader
    batch_size = 256
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
    testloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False, num_workers=16)  # create your dataloader
    model.eval()

    # test
    feature_vector = []
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(testloader):
            if batch_idx % 5 == 0:
                print("Extracting features: {}/{}".format(batch_idx, int(len(my_dataset) / batch_size) + 1))
            # print(inputs.shape)
            inputs = inputs.to(device)
            h = model(inputs)
            h = h.squeeze()
            h = h.detach()
            feature_vector.extend(h.cpu().detach().numpy())

    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector


def test_rank_selection(x, y, x_candidate, y_candidate, model, budgets, args="", data_type="mnist"):
    x_test = np.concatenate((x, x_candidate))
    y_test = np.concatenate((y, y_candidate))
    if data_type == "mnist":
        latents = x_test.reshape(-1, 784)
    elif data_type == "cifar10":
        feature_extractor_net = models.resnet18(pretrained=False).to(device)
        feature_extractor_net.load_state_dict(torch.load('/home/qhu/projects/TSattack/metrics/TestRank/ckpt_byol/cifar10_fc_224.pt', map_location=device))
        feature_extractor_net = nn.Sequential(*list(feature_extractor_net.children())[:-1])
        latents = get_features(x_test, y_test, feature_extractor_net)
    elif data_type == "traffic":
        latents = x_test.reshape(-1, 3072)[:, :1000]
    elif data_type == "svhn":
        latents = x_test.reshape(-1, 3072)[:, :1000]
    else:
        latents = x_test.reshape(-1, 3072)[:, :1000]
    # calculate right and wrong, labeled and unlabeled data
    st = time.time()
    no_neighbors = 100
    labeled_indices = np.arange(len(x))
    unlabeled_indices = np.arange(len(x_candidate)) + len(x)
    correct_array, logits = model_eval(model, x_test, y_test)
    misclass_array = (correct_array == 0).astype(int)

    # correct - neg
    # wrong - pos
    neg_case_indexes = []
    pos_case_indexes = []
    # print("Test the labeled samples")
    labled_neg_indices = list(labeled_indices[np.nonzero(correct_array[labeled_indices])[0]])
    labled_pos_indices = list(labeled_indices[np.nonzero(misclass_array[labeled_indices])[0]])
    neg_case_indexes += labled_neg_indices
    pos_case_indexes += labled_pos_indices
    selected = set()

    # init some saving list
    budget_lst = []
    pfd_lst = []
    ideal_pfd_lst = []
    budget_lst.insert(0, 0)
    pfd_lst.insert(0, 0)
    ideal_pfd_lst.insert(0, 0)
    # Iterative
    # print("# of unlabeled test inputs: {}".format(len(x_candidate)))

    # get classification result rank
    prob = softmax(logits, axis=1)
    confidence = np.sum(np.square(prob), axis=1)
    prob = torch.from_numpy(prob)

    # main method: apply GNN classification algorithm
    hidden_dim = default_hyperparams['custom']['gnn_hidden_dim']
    epochs = default_hyperparams['custom']['gnn_train_epoch']
    # create dataset
    # print('Start runing GNN classification algorithm')

    # approximate version
    # GNN training
    x_l_indices = labeled_indices
    x_u_indices = unlabeled_indices
    x_l = torch.from_numpy(latents[x_l_indices]).float().to(device)
    x_u = torch.from_numpy(latents[x_u_indices]).float().to(device)

    batch = torch.tensor([0 for _ in range(x_l.shape[0])]).to(device)
    edge_index_t = my_knn_graph(x_l, x_l, batch_x=batch, batch_y=batch, cosine=False, loop=False, k=no_neighbors)
    # print("l-2-l edge index: ", edge_index_t)
    # edge_index = torch.zeros_like(edge_index_t)
    new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
    new_edge_index_l1 = [x_l_indices[i] for i in list(edge_index_t[1])]
    l2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
    # print("replaced edge index: ", l2l_edge_index)

    batch_x = torch.tensor([0 for _ in range(x_l.shape[0])]).to(device)
    batch_y = torch.tensor([0 for _ in range(x_u.shape[0])]).to(device)
    edge_index_t = my_knn_graph(x_l, x_u, batch_x=batch_x, batch_y=batch_y, cosine=False, loop=False, k=no_neighbors)
    # print("u-2-l edge index: ", edge_index_t)
    new_edge_index_l0 = [x_l_indices[i] for i in list(edge_index_t[0])]
    new_edge_index_l1 = [x_u_indices[i] for i in list(edge_index_t[1])]
    u2l_edge_index = torch.tensor([new_edge_index_l0, new_edge_index_l1])
    # print("replaced edge index: ", u2l_edge_index)

    edge_index = torch.cat([l2l_edge_index, u2l_edge_index], dim=1)
    # print("final edge_index, ", u2l_edge_index)
    # print(
    #     "Finish calculate edge index, the shape is {}, time cost: {:4f}".format(edge_index.shape, time.time() - st))
    # end approximation

    edge_weight = torch.ones((edge_index.size(1),), device=edge_index.device)
    for i in range(edge_index.size(1)):
        edge_weight[i] = distance.cosine(latents[edge_index[0, i]], latents[edge_index[1, i]])
    # print('Finish calculating edge weight. example of edge_weight: {}'.format(edge_weight[:10]))
    # print("time cost to calculate edge weight: {}".format(time.time()-new_time), log)
    # return

    y = torch.zeros((latents.shape[0]), dtype=torch.long)
    imbalance_ratio = 1. * len(neg_case_indexes) / len(pos_case_indexes)
    # print("neg:pos imbalance ratio: {}".format(imbalance_ratio))


    # importance sampling
    class_weights = torch.tensor([1. / (1 + imbalance_ratio), 1. * imbalance_ratio / (1 + imbalance_ratio)])
    # print("class_weights: {}".format(class_weights), log)

    y[pos_case_indexes] = 1
    y[neg_case_indexes] = 0
    # print('positives: {}'.format(len(pos_case_indexes)))
    # print('negatives: {}'.format(len(neg_case_indexes)))

    dataset = Data(x=torch.from_numpy(latents).float(), y=y, edge_index=edge_index)
    dataset.edge_weight = edge_weight
    # print('example edge index: {}'.format(dataset.edge_index))
    # print('example edge index: max {} min {}'.format(dataset.edge_index[0].max(), dataset.edge_index[0].min()))
    # print ('example x: {}'.format(dataset.x[:10]))
    # print('example y: {}'.format(dataset.y.sum()))
    #
    # print("dataset info: {}".format(dataset))
    dataset.num_classes = y.max().item() + 1
    # print("**number of classes: {}".format(dataset.num_classes))

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
    # print("GNN training info")
    # print("number of train cases: {}".format(dataset.train_mask.sum().item()))
    # print("number of val cases: {}".format(dataset.val_mask.sum().item()))
    # print("number of test cases: {}".format(dataset.test_mask.sum().item()))
    # print("GNN input dimension: {}".format(dataset.num_node_features))
    # print("GNN output dimension: {}".format(dataset.num_classes))

    # build gnn model
    gcn_model = GNNStack(input_dim=max(dataset.num_node_features, 1),
                         hidden_dim=hidden_dim,
                         output_dim=dataset.num_classes)
    opt = optim.Adam(gcn_model.parameters(), lr=0.001, weight_decay=5e-4)

    # build mlp model
    # if args.learn_mixed:
    # print('shape of sample embedding: ', hidden_dim + prob.shape[1] + 1)
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

    gcn_model.eval()
    with torch.no_grad():
        emb, output_distribution = gcn_model(dataset.x, dataset.edge_index, edge_weight=dataset.edge_weight)
        output_distribution = torch.exp(output_distribution)
        # print("output distribution: {}".format(output_distribution.shape), log=log)
    output_distribution = output_distribution.cpu().numpy()

    # print("Mixed method enabled: combine gini and correlation based method, learn based")
    # print("dim: {}/{}/{}".format(emb.shape, torch.tensor(confidence).unsqueeze(1).shape, prob.shape))

    sample_emd = torch.cat([emb.to(device), torch.tensor(confidence).unsqueeze(1).to(device), prob.to(device)],
                           dim=1).type(torch.FloatTensor)
    # bf_mixed

    # output_distribution = output_distribution.detach().numpy()
    mix_rank_indicator = output_distribution[:, 1] * (1 - confidence)
    ranked_indexes = np.argsort(mix_rank_indicator)[::-1].astype(np.int64)

    # learn_mixed
    # outputs = mlp_model(sample_emd.to(device))
    # output_distribution = F.softmax(outputs.cpu(), dim=1)
    # output_distribution = output_distribution.detach().numpy()
    # ranked_indexes = np.argsort(output_distribution[:, 1])[::-1].astype(np.int64)

    #  nothing:
    # ranked_indexes = np.argsort(output_distribution[:, 1])[::-1].astype(np.int64)
    index2select = [i for i in ranked_indexes if ((i not in selected) and (i not in labeled_indices))]
    # print("ranked output distri for selection: {}".format(output_distribution[:, 1][index2select][:100]))

    # p_budget = budget
    selected_index_ls = []
    for budget in budgets:
        # print("\n ###### budget percent is {}% ######".format(p_budget))
        selected_temp = copy.deepcopy(selected)
        # model2test_temp = copy.deepcopy(model2test)
        neg_case_indexes_temp = copy.deepcopy(neg_case_indexes)
        pos_case_indexes_temp = copy.deepcopy(pos_case_indexes)

        # budget = int(p_budget * len(x_candidate) / 100.0)

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

        selected_index = np.array(list(selected_temp)) - len(x)
        selected_index_ls.append(selected_index)
    return selected_index_ls


if __name__ == "__main__":
    # (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train.astype("float32") / 255
    #
    # x_train = x_train.reshape(-1, 28, 28, 1)
    # x_test = np.load("../../datasets/mnist/RT_test_x.npy")
    # y_test = np.load("../../datasets/mnist/RT_test_y.npy")
    # x_test = x_test.astype("float32") / 255
    # x_test = x_test.reshape(-1, 28, 28, 1)
    # # y_train = tf.keras.utils.to_categorical(y_train, 10)
    # # y_test = tf.keras.utils.to_categorical(y_test, 10)
    # model = tf.keras.models.load_model("../../models/mnist/lenet5.h5")
    # # test_rank_prior(x_train[:10000], y_train[:10000], x_test, y_test, model, "", "args")
    # test_rank_selection(x_train[:1000], y_train[:1000], x_test[:1000], y_test[:1000], model, 100)
    (x_train, y_train), (x_final, y_final) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    y_final = y_final.reshape(10000, )
    x_final = x_final.astype("float32") / 255
    x_final -= x_train_mean
    x_train -= x_train_mean
    y_train = y_train.reshape(-1, )
    y_final = y_final.reshape(-1, )
    model = tf.keras.models.load_model("../../models/cifar10/resnet20.h5")
    test_rank_selection(x_train[:1000], y_train[:1000], x_final[:1000], y_final[:1000], model, 100, data_type="cifar10")


