import re
import torch
import numpy as np
from collections import defaultdict
from functools import wraps
from rocksdict import Rdict
import cloudpickle
import numpy as np
import os

path = os.path.dirname(os.path.abspath(__file__))  # 项目根目录
caches = Rdict(path+"/temp/cache")
CACHE_SWITCH = True


def cache_with_callable(key, callable):
    if key in caches and CACHE_SWITCH:
        print(f"key:{key} is in caches")
        return caches[key]
    else:
        print(f"key:{key} is not in caches")
        value = callable()
        try:
            caches[key] = value
        except Exception as e:
            print(e)
        return value


def rocks_cache(func):
    @wraps(func)
    def wrap(*args):
        pickle_args = cloudpickle.dumps(args)
        if pickle_args not in caches:
            result = func(*args)
            caches[pickle_args] = cloudpickle.dumps(result)
        return cloudpickle.loads(caches[pickle_args])
    return wrap


def macro_f1(pred, targ, num_classes=None):
    pred = torch.max(pred, 1)[1]
    tp_out = []
    fp_out = []
    fn_out = []
    if num_classes is None:
        num_classes = sorted(set(targ.cpu().numpy().tolist()))
    else:
        num_classes = range(num_classes)
    for i in num_classes:
        tp = ((pred == i) & (targ == i)).sum().item()  # 预测为i，且标签的确为i的
        fp = ((pred == i) & (targ != i)).sum().item()  # 预测为i，但标签不是为i的
        fn = ((pred != i) & (targ == i)).sum().item()  # 预测不是i，但标签是i的
        tp_out.append(tp)
        fp_out.append(fp)
        fn_out.append(fn)

    eval_tp = np.array(tp_out)
    eval_fp = np.array(fp_out)
    eval_fn = np.array(fn_out)

    precision = eval_tp / (eval_tp + eval_fp)
    precision[np.isnan(precision)] = 0
    precision = np.mean(precision)

    recall = eval_tp / (eval_tp + eval_fn)
    recall[np.isnan(recall)] = 0
    recall = np.mean(recall)

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


def accuracy(pred, targ):
    pred = torch.max(pred, 1)[1]
    acc = ((pred == targ).float()).sum().item() / targ.size()[0]

    return acc


def print_graph_detail(graph):
    """
    格式化显示Graph参数
    :param graph:
    :return:
    """
    import networkx as nx
    dst = {"nodes": nx.number_of_nodes(graph),
           "edges": nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates": nx.number_of_isolates(graph),
           "覆盖度": 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    print_table(dst)


def print_table(dst):
    table_title = list(dst.keys())
    from prettytable import PrettyTable
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    print(table)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = "hdd_data/prepare_dataset/model/model.pt"

    def __call__(self, val_loss, model=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        return torch.load(self.model_path)


class LogResult:
    def __init__(self):
        self.result = defaultdict(list)
        pass

    def log(self, result: dict):
        for key, value in result.items():
            self.result[key].append(value)

    def log_single(self, key, value):
        self.result[key].append(value)

    def show_str(self):
        print()
        string = ""
        for key, value_lst in self.result.items():
            value = np.mean(value_lst)
            if isinstance(value, int):
                string += f"{key}:\n{value}\n{max(value_lst)}\n{min(value_lst)}\n"
            else:
                string += f"{key}:\n{value:.4f}\n{max(value_lst):.4f}\n{min(value_lst):.4f} \n"
        print(string)