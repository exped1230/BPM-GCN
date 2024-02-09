import numpy as np
import pickle
import torch
from torch.utils.data import Dataset
import sys
import math
sys.path.extend(['../'])
from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_m_path, data_p_path, label_path,feature_path,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        """
        
        :param data_path: 
        :param label_path: 
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move: 
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        """

        self.debug = debug
        self.data_m_path = data_m_path
        self.data_p_path = data_p_path
        self.label_path = label_path
        self.feature_path=feature_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M

        if '.npy' not in self.label_path:
            try:
                with open(self.label_path) as f:
                    self.sample_name, self.label = pickle.load(f)
            except:
                # for pickle file from python2
                with open(self.label_path, 'rb') as f:
                    self.sample_name, self.label = pickle.load(f, encoding='latin1')
        else:
            self.label = np.load(self.label_path, allow_pickle=True)
            self.sample_name = None
        # load data
        if self.use_mmap:
            self.data_m = np.load(self.data_m_path, mmap_mode='r')
            self.data_p = np.load(self.data_p_path, mmap_mode='r')
            self.feature=np.load(self.feature_path, mmap_mode='r')
        else:
            self.data_m = np.load(self.data_m_path)
            self.data_p = np.load(self.data_p_path)
            self.feature=np.load(self.feature_path)
        if self.debug:
            self.label = self.label[0:100]
            self.data_m = self.data_m[0:100]
            self.data_p = self.data_p[0:100]
            self.sample_name = self.sample_name[0:100]
            self.feature=self.feature[0:100]

    def get_mean_map(self):
        data_m = self.data_m
        N, C, T, V, M = data_m.shape
        # self.mean_map_m = data_m.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)+1e-6
        # self.std_map_m = data_m.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))+1e-6
        self.mean_map_m = data_m.mean(axis=0)
        self.std_map_m = data_m.std(axis=0)+1e-6

        data_p = self.data_p
        N, C, T, V, M = data_p.shape
        # self.mean_map_p = data_p.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)+1e-6
        # self.std_map_p = data_p.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))+1e-6
        self.mean_map_p=data_p.mean(axis=0)
        self.std_map_p=data_p.std(axis=0)+1e-6

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy_m = self.data_m[index]
        data_numpy_p = self.data_p[index]

        label = self.label[index]
        data_numpy_m = np.array(data_numpy_m)
        data_numpy_p = np.array(data_numpy_p)
        feature_numpy=self.feature[index]
        feature_numpy=np.array(feature_numpy)

        if self.normalization:
            data_numpy_m = (data_numpy_m - self.mean_map_m) / self.std_map_m
            data_numpy_p = (data_numpy_p - self.mean_map_p) / self.std_map_p
        if self.random_shift:
            data_numpy_m = tools.random_shift(data_numpy_m)
            data_numpy_p = tools.random_shift(data_numpy_p)
        if self.random_choose:
            data_numpy_m = tools.random_choose(data_numpy_m, self.window_size)
            data_numpy_p = tools.random_choose(data_numpy_p, self.window_size)
        elif self.window_size > 0:
            data_numpy_m = tools.auto_pading(data_numpy_m, self.window_size)
            data_numpy_p = tools.auto_pading(data_numpy_p, self.window_size)
        if self.random_move:
            data_numpy_m = tools.random_move(data_numpy_m)
            data_numpy_p = tools.random_move(data_numpy_p)
        return data_numpy_m, data_numpy_p, label,feature_numpy, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    

class FeederDataset(Dataset):
    def __init__(self, data_p, data_m, feature, label, random_choose=False, random_shift=False, random_move=False, window_size=-1, normalization=False, debug=False, use_mmap=True):
        self.data_p = data_p
        self.data_m = data_m
        self.feature = feature
        self.label = label
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy_m = self.data_m[index]
        data_numpy_p = self.data_p[index]

        label = self.label[index]
        data_numpy_m = np.array(data_numpy_m)
        data_numpy_p = np.array(data_numpy_p)
        feature_numpy = self.feature[index]
        feature_numpy = np.array(feature_numpy)

        if self.normalization:
            data_numpy_m = (data_numpy_m - self.mean_map_m) / self.std_map_m
            data_numpy_p = (data_numpy_p - self.mean_map_p) / self.std_map_p
        if self.random_shift:
            data_numpy_m = tools.random_shift(data_numpy_m)
            data_numpy_p = tools.random_shift(data_numpy_p)
        if self.random_choose:
            data_numpy_m = tools.random_choose(data_numpy_m, self.window_size)
            data_numpy_p = tools.random_choose(data_numpy_p, self.window_size)
        elif self.window_size > 0:
            data_numpy_m = tools.auto_pading(data_numpy_m, self.window_size)
            data_numpy_p = tools.auto_pading(data_numpy_p, self.window_size)
        if self.random_move:
            data_numpy_m = tools.random_move(data_numpy_m)
            data_numpy_p = tools.random_move(data_numpy_p)
        return data_numpy_m, data_numpy_p, label,feature_numpy, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
    

class FeederSplit(Dataset):
    def __init__(self, train_data_m_path, train_data_p_path, train_label_path, train_feature_path,
                 test_data_m_path, test_data_p_path, test_label_path, test_feature_path,
                 train_ratio=0.9, val_ratio=0, test_ratio=0.1,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True):
        self.debug = debug
        self.train_data_m_path = train_data_m_path
        self.train_data_p_path = train_data_p_path
        self.train_label_path = train_label_path
        self.train_feature_path = train_feature_path
        self.test_data_m_path = test_data_m_path
        self.test_data_p_path = test_data_p_path
        self.test_label_path = test_label_path
        self.test_feature_path = test_feature_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.load_data()
        if normalization:
            self.get_mean_map()
        
        length = len(self.label)
        idxes = np.arange(length)
        train_length = int(length * train_ratio)
        # test_length = length - train_length
        train_idxes = np.random.choice(idxes, train_length, replace=False)
        test_idxes = np.delete(idxes, train_idxes)
        if train_ratio + test_ratio != 1: test_idxes = np.random.choice(test_idxes, int(len(test_idxes)*test_ratio/(1-train_ratio)), replace=False)
        self.train_set = FeederDataset(self.data_p[train_idxes], self.data_m[train_idxes], self.feature[train_idxes], self.label[train_idxes], self.random_choose, self.random_shift, self.random_move, self.window_size, self.normalization, self.debug, self.use_mmap)
        self.test_set = FeederDataset(self.data_p[test_idxes], self.data_m[test_idxes], self.feature[test_idxes], self.label[test_idxes], self.random_choose, self.random_shift, self.random_move, self.window_size, self.normalization, self.debug, self.use_mmap)

    def get_data(self):
        return self.train_set, self.test_set

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.train_label_path) as f:
                self.train_sample_name, self.train_label = pickle.load(f)
            with open(self.test_label_path) as f:
                self.test_sample_name, self.test_label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.train_label_path, 'rb') as f:
                self.train_sample_name, self.train_label = pickle.load(f, encoding='latin1')
            with open(self.test_label_path, 'rb') as f:
                self.test_sample_name, self.test_label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.train_data_m = np.load(self.train_data_m_path, mmap_mode='r')
            self.train_data_p = np.load(self.train_data_p_path, mmap_mode='r')
            self.train_feature=np.load(self.train_feature_path, mmap_mode='r')
            self.test_data_m = np.load(self.test_data_m_path, mmap_mode='r')
            self.test_data_p = np.load(self.test_data_p_path, mmap_mode='r')
            self.test_feature=np.load(self.test_feature_path, mmap_mode='r')
        else:
            self.train_data_m = np.load(self.train_data_m_path)
            self.train_data_p = np.load(self.train_data_p_path)
            self.train_feature=np.load(self.train_feature_path)
            self.test_data_m = np.load(self.test_data_m_path)
            self.test_data_p = np.load(self.test_data_p_path)
            self.test_feature=np.load(self.test_feature_path)
        
        self.train_label.extend(self.test_label)
        self.label = np.array(self.train_label)
        self.data_p = np.concatenate([self.train_data_p, self.test_data_p], axis=0)
        self.data_m = np.concatenate([self.train_data_m, self.test_data_m], axis=0)
        self.feature = np.concatenate([self.train_feature, self.test_feature], axis=0)

        if self.debug:
            self.label = self.label[0:100]
            self.data_m = self.data_m[0:100]
            self.data_p = self.data_p[0:100]
            self.sample_name = self.sample_name[0:100]
            self.feature=self.feature[0:100]

    def get_mean_map(self):
        data_m = self.data_m
        N, C, T, V, M = data_m.shape
        # self.mean_map_m = data_m.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)+1e-6
        # self.std_map_m = data_m.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))+1e-6
        self.mean_map_m = data_m.mean(axis=0)
        self.std_map_m = data_m.std(axis=0)+1e-6

        data_p = self.data_p
        N, C, T, V, M = data_p.shape
        # self.mean_map_p = data_p.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)+1e-6
        # self.std_map_p = data_p.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))+1e-6
        self.mean_map_p=data_p.mean(axis=0)
        self.std_map_p=data_p.std(axis=0)+1e-6


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def test(data_path, label_path, vid=None, graph=None, is_3d=False):
    '''
    vis the samples using matplotlib
    :param data_path: 
    :param label_path: 
    :param vid: the id of sample
    :param graph: 
    :param is_3d: when vis NTU, set it True
    :return: 
    '''
    import matplotlib.pyplot as plt
    loader = torch.utils.data.DataLoader(
        dataset=Feeder(data_path, label_path),
        batch_size=64,
        shuffle=False,
        num_workers=2)

    if vid is not None:
        sample_name = loader.dataset.sample_name
        sample_id = [name.split('.')[0] for name in sample_name]
        index = sample_id.index(vid)
        data, label, index = loader.dataset[index]
        data = data.reshape((1,) + data.shape)

        # for batch_idx, (data, label) in enumerate(loader):
        N, C, T, V, M = data.shape

        plt.ion()
        fig = plt.figure()
        if is_3d:
            from mpl_toolkits.mplot3d import Axes3D
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        if graph is None:
            p_type = ['b.', 'g.', 'r.', 'c.', 'm.', 'y.', 'k.', 'k.', 'k.', 'k.']
            pose = [
                ax.plot(np.zeros(V), np.zeros(V), p_type[m])[0] for m in range(M)
            ]
            ax.axis([-1, 1, -1, 1])
            for t in range(T):
                for m in range(M):
                    pose[m].set_xdata(data[0, 0, t, :, m])
                    pose[m].set_ydata(data[0, 1, t, :, m])
                fig.canvas.draw()
                plt.pause(0.001)
        else:
            p_type = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'k-', 'k-', 'k-']
            import sys
            from os import path
            sys.path.append(
                path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
            G = import_class(graph)()
            edge = G.inward
            pose = []
            for m in range(M):
                a = []
                for i in range(len(edge)):
                    if is_3d:
                        a.append(ax.plot(np.zeros(3), np.zeros(3), p_type[m])[0])
                    else:
                        a.append(ax.plot(np.zeros(2), np.zeros(2), p_type[m])[0])
                pose.append(a)
            ax.axis([-1, 1, -1, 1])
            if is_3d:
                ax.set_zlim3d(-1, 1)
            for t in range(T):
                for m in range(M):
                    for i, (v1, v2) in enumerate(edge):
                        x1 = data[0, :2, t, v1, m]
                        x2 = data[0, :2, t, v2, m]
                        if (x1.sum() != 0 and x2.sum() != 0) or v1 == 1 or v2 == 1:
                            pose[m][i].set_xdata(data[0, 0, t, [v1, v2], m])
                            pose[m][i].set_ydata(data[0, 1, t, [v1, v2], m])
                            if is_3d:
                                pose[m][i].set_3d_properties(data[0, 2, t, [v1, v2], m])
                fig.canvas.draw()
                # plt.savefig('/home/lshi/Desktop/skeleton_sequence/' + str(t) + '.jpg')
                plt.pause(0.01)


if __name__ == '__main__':
    import os

    os.environ['DISPLAY'] = 'localhost:10.0'
    data_path = "../data/ntu/xview/val_data_joint.npy"
    label_path = "../data/ntu/xview/val_label.pkl"
    graph = 'graph.ntu_rgb_d.Graph'
    test(data_path, label_path, vid='S004C001P003R001A032', graph=graph, is_3d=True)
    # data_path = "../data/kinetics/val_data.npy"
    # label_path = "../data/kinetics/val_label.pkl"
    # graph = 'graph.Kinetics'
    # test(data_path, label_path, vid='UOD7oll3Kqo', graph=graph)
