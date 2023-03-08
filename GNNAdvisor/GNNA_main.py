import sys
import time
import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from tqdm import *
from scipy.sparse import *

import GNNAdvisor as GNNA           # import GNNAdvisor
import pubmed_util
from gnn_conv import *
from dataset import *
import datetime

#torch.set_printoptions(edgeitems=3703/2)
parser = argparse.ArgumentParser()
# Dataset related parameters.
parser.add_argument("--dataDir", type=str, default="../osdi-ae-graphs", help="the path to graphs")
parser.add_argument("--dataset", type=str, default='amazon0601', help="dataset")
parser.add_argument("--dim", type=int, default=96, help="input embedding dimension size")
parser.add_argument("--hidden", type=int, default=16, help="hidden dimension size")
parser.add_argument("--classes", type=int, default=22, help="output classes size")

# Model training related parameters.
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gin'],  help="GCN or GIN")
parser.add_argument("--num_epoches", type=int, default=400, help="number of epoches for training, default=200")

# Manually set the performance related parameters
parser.add_argument("--partSize", type=int, default=32, help="neighbor-group size")
parser.add_argument("--dimWorker", type=int, default=32, help="number of worker threads (MUST < 32)")
parser.add_argument("--warpPerBlock", type=int, default=4, help="number of warp per block, recommended: GCN: 8, GIN: 2")
parser.add_argument("--sharedMem", type=int, default=100, help="shared memory size of each block (Quadro P6000 64(KB) sm_61), default=100(KB) for RTX3090 sm_86")

# Additional flags for studies.
parser.add_argument('--manual_mode', type=str, choices=['True', 'False'], default='True', help="True: use manual config, False: auto config, default: True")
parser.add_argument('--verbose_mode', type=str, choices=['True', 'False'], default='False', help="True: verbose mode, False: simple mode, default: False")
parser.add_argument('--enable_rabbit', type=str, choices=['True', 'False'], default='False', help="True: enable rabbit reordering, False, disable rabbit reordering, default: False (disable for both manual and auto mode).")
parser.add_argument('--loadFromTxt', type=str, choices=['True', 'False'], default='False', help="True: load the graph TXT edge list, False: load from .npy, default: False (load from npz fast)")
parser.add_argument('--single_spmm', type=str, choices=['True', 'False'], default='False', help="True: profile the single SpMM (neighbor aggregation) kernel for number epoches times")
parser.add_argument('--verify_spmm', type=str, choices=['True', 'False'], default='False', help="True: verify the output correctness of a single SpMM (neighbor aggregation) kernel against the CPU reference implementation.")

args = parser.parse_args()
print(args)

partSize, dimWorker, warpPerBlock, sharedMem = args.partSize, args.dimWorker, args.warpPerBlock, args.sharedMem
manual_mode = args.manual_mode == 'True'
verbose_mode = args.verbose_mode == 'True'
enable_rabbit = args.enable_rabbit == 'True'
loadFromTxt = args.loadFromTxt == 'True'
single_spmm = args.single_spmm == 'True'
verify_spmm = args.verify_spmm == 'True'

# requires GPU for evaluation.
assert torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

####################################
# loading data from files
####################################
if loadFromTxt:
    #path = osp.join(args.dataDir, args.dataset)
    print("load from txt")

    #dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=True, verbose=verbose_mode)
    #path = "/mnt/huge_26TB/data/test2/reddit/graph_structure/reddit_graph_undirect.txt"
    #dim = 602
    #classes = 41
    #path = "/mnt/huge_26TB/data/test2/citeseer/graph_structure/citeseer_graph_undirect.txt"
    #path = "/mnt/huge_26TB/data/test2/citeseer/graph_structure"
    if args.dataset == 'citeseer':
        path = "/mnt/huge_26TB/data/test2/citeseer/graph_structure/" #citeseer_graph_undirect.txt"
        dim = 3703
        classes = 6
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/citeseer/feature/citeseer_feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/citeseer/index/citeseer_train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/citeseer/label/citeseer_y_label.txt")
        
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/citeseer/index/citeseer_test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/citeseer/label/citeseer_test_y_label.txt")

    elif args.dataset == 'cora':
        path = "/mnt/huge_26TB/data/test2/cora/graph_structure/" #cora.txt"
        dim = 1433
        classes = 7
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/cora/feature/cora_feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/cora/index/cora_train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/cora/label/cora_y_label.txt")
        
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/cora/index/cora_test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/cora/label/cora_test_y_label.txt")

    elif args.dataset == 'pubmed':
        path = "/mnt/huge_26TB/data/test2/pubmed/graph_structure/" #pubmed_graph_undouble.txt"
        dim = 500
        classes = 3
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/pubmed/feature/feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/pubmed/index/train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/pubmed/label/y_label.txt")
        
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/pubmed/index/test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/pubmed/label/test_y_label.txt")


    elif args.dataset == 'reddit':
        path = "/mnt/huge_26TB/data/test2/reddit/graph_structure/" #reddit_graph_undirect.txt"
        dim = 602
        classes = 41
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/reddit/feature/reddit_feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/reddit/index/reddit_train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/reddit/label/reddit_y_label.txt")
        
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/reddit/index/reddit_test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/reddit/label/reddit_test_y_label.txt")

    elif args.dataset == 'ogb-product':
        path = "/mnt/huge_26TB/data/test2/ogb-product/graph_structure/" #graph_undirect.txt"
        dim = 100
        classes = 47
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/ogb-product/feature/feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/ogb-product/index/train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/ogb-product/label/train_y_label.txt")
        
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/ogb-product/index/test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/ogb-product/label/test_y_label.txt")
    
    elif args.dataset == 'ogb-arx':
        path = "/mnt/huge_26TB/data/test2/ogb-arx/graph_structure/" #graph_undirect.txt"
        dim = 128
        classes = 40
        dataset = custom_dataset(path, dim, classes, load_from_txt=True, verbose=verbose_mode)
        feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/ogb-arx/feature/feature.txt")
        feat = torch.tensor(feat)
        feat = feat.to(device)
        #print("feat from loadtxt", feat)
        train_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/ogb-arx/index/train_index.txt")
        train_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/ogb-arx/label/train_y_label.txt")



    train_id = torch.tensor(train_id)
    train_y_label = torch.tensor(train_y_label)
    train_id=train_id.to(device)
    train_y_label=train_y_label.to(device)


else:
    path = osp.join(args.dataDir, args.dataset+".npz")
    dataset = custom_dataset(path, args.dim, args.classes, load_from_txt=False, verbose=verbose_mode)
    feat = pubmed_util.read_feature_info("/mnt/huge_26TB/data/test2/citeseer/feature/citeseer_feature.txt")
    feat = torch.tensor(feat)
    feat = feat.to(device)

num_nodes = dataset.num_nodes
num_edges = dataset.num_edges
column_index = dataset.column_index
row_pointers = dataset.row_pointers
degrees = dataset.degrees

####################################
# Building input property profile.
####################################
inputInfo = inputProperty(row_pointers, column_index, degrees, 
                            partSize, dimWorker, warpPerBlock, sharedMem,
                            hiddenDim=args.hidden, dataset_obj=dataset, enable_rabbit=enable_rabbit,
                            manual_mode=manual_mode, verbose=verbose_mode)

####################################
# Decider for parameter selection.
####################################
inputInfo.decider()

inputInfo = inputInfo.set_input()
if verbose_mode:
    print('----------------------------')
    inputInfo.print_param()
    print()

inputInfo = inputInfo.set_hidden()
if verbose_mode:
    inputInfo.print_param()
    print()
    print('----------------------------')
# sys.exit(0)

####################################
# Building neighbor partitioning.
####################################
inputInfo.print_param()
dataset.avg_degree
start = time.perf_counter()
partPtr, part2Node = GNNA.build_part(inputInfo.partSize, inputInfo.row_pointers)
build_neighbor_parts = time.perf_counter() - start
if verbose_mode:
    print("# Build nb_part (s): {:.3f}".format(build_neighbor_parts))

inputInfo.row_pointers  = inputInfo.row_pointers.to(device)
inputInfo.column_index  = inputInfo.column_index.to(device)
inputInfo.partPtr = partPtr.int().to(device)
inputInfo.part2Node  = part2Node.int().to(device)

####################################
# Verifing a single SpMM kernel
# against the CPU reference.
####################################
if verify_spmm:
    from unitest import *
    valid = Verification(args.hidden, \
                        inputInfo.row_pointers, inputInfo.column_index, degrees, \
                        inputInfo.partPtr, inputInfo.part2Node, \
                        partSize, dimWorker, warpPerBlock)
    valid.compute()
    valid.reference(dataset.edge_index, dataset.val, dataset.num_nodes)
    valid.compare()
    sys.exit(0)

####################################
# Profiling a single SpMM kernel
####################################
if single_spmm:
    from unitest import *
    valid = Verification(args.hidden, \
                        inputInfo.row_pointers, inputInfo.column_index, degrees, \
                        inputInfo.partPtr, inputInfo.part2Node, \
                        partSize, dimWorker, warpPerBlock)
    valid.profile_spmm(round=args.num_epoches)
    sys.exit(0)

####################################
# Building GNN model
####################################
if args.model == 'gcn':
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GCNConv(dataset.num_features, args.hidden)
            #print("conv1 layer num_feas, num_hidden", dataset.num_features, args.hidden)
            self.conv2 = GCNConv(args.hidden, dataset.num_classes)
            #print("conv2 layer num_feas, num_hidden", args.hidden, dataset.num_classes)

        def forward(self):
            #x = dataset.x
            x = feat
            #print("x from forward", x.size())
            x = F.relu(self.conv1(x, inputInfo.set_input()))
            x = self.conv2(x, inputInfo.set_hidden())
            return F.log_softmax(x, dim=1)
else:
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = GINConv(dataset.num_features, args.hidden)
            self.conv2 = GINConv(args.hidden, args.hidden)
            self.conv3 = GINConv(args.hidden, args.hidden)
            self.conv4 = GINConv(args.hidden, args.hidden)
            self.conv5 = GINConv(args.hidden, dataset.num_classes)

        def forward(self):
            #x = dataset.x
            x = feat
            x = F.relu(self.conv1(x, inputInfo.set_input()))
            x = F.relu(self.conv2(x, inputInfo.set_hidden()))
            x = F.relu(self.conv3(x, inputInfo.set_hidden()))
            x = F.relu(self.conv4(x, inputInfo.set_hidden()))
            x = self.conv5(x, inputInfo.set_hidden())
            return F.log_softmax(x, dim=1)

model, dataset = Net().to(device), dataset.to(device)
if verbose_mode:
    print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

####################################
# Define training function.
####################################

def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[:200], dataset.y[:200])
    print("loss", loss)
    loss.backward()
    optimizer.step()
def train_txt():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[train_id], train_y_label)
    #print("loss", loss)
    #loss = F.nll_loss(model()[:], dataset.y[:])
    loss.backward()
    optimizer.step()

if __name__ == '__main__':
    # dry run
    #for _ in range(2):
        #train()
    # exit(0)
    """
    if args.dataset == 'citeseer':
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/citeseer/index/citeseer_test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/citeseer/label/citeseer_test_y_label.txt")
    elif args.dataset == 'cora':
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/cora/index/cora_test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/cora/label/cora_test_y_label.txt")
    elif args.dataset == 'pubmed':
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/pubmed/index/test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/pubmed/label/test_y_label.txt")
    elif args.dataset == 'ogb-arx':
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/ogb-arx/index/test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/ogb-arx/label/test_y_label.txt")
    elif args.dataset == 'reddit':
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/reddit/index/reddit_test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/reddit/label/reddit_test_y_label.txt")
    elif args.dataset == 'ogb-product':
        test_id = pubmed_util.read_index_info("/mnt/huge_26TB/data/test2/ogb-product/index/test_index.txt")
        test_y_label =  pubmed_util.read_label_info("/mnt/huge_26TB/data/test2/ogb-product/label/test_y_label.txt")
    """
    
    test_id = torch.tensor(test_id)
    test_y_label = torch.tensor(test_y_label)
    test_id = test_id.to(device)
    test_y_label = test_y_label.to(device)

    torch.cuda.synchronize()
    start_train = time.perf_counter()
    for _ in tqdm(range(1, args.num_epoches + 1)):
        if loadFromTxt:
            start = datetime.datetime.now()
            train_txt()
            end = datetime.datetime.now()
            difference = end - start

            acc_val = pubmed_util.accuracy(model()[test_id], test_y_label)
            print('Epoch %d | Test_accuracy: %.4f' % (args.num_epoches + 1, acc_val))
            #print("the time of advisor is:", difference)
        else:
            train()

    torch.cuda.synchronize()
    train_time = time.perf_counter() - start_train
    #check the accuracy
    print("data", args.dataset)
    
    print("prediction size", model()[test_id].size())
    print("label size", test_y_label.size())
    #acc_val = pubmed_util.accuracy(model()[300:400], dataset.y[300:400])

    acc_val = pubmed_util.accuracy(model()[test_id], test_y_label)
    print('Epoch %d | Test_accuracy: %.4f' % (args.num_epoches + 1, acc_val))

    print('Time (second for all epoches): {:.3f}'.format(train_time))
    print('num_epochs', args.num_epoches, args.model)
