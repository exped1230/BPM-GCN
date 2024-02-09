#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm
import time
import torch.nn.functional as F


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/EGait_journal/train_diff_combine_double_score_fagg.yaml',
        # default='./config/kinetics-skeleton/train_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 2],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=20,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')
    
    parser.add_argument('--train_ratio', default=0.9)
    parser.add_argument('--val_ratio', default=0.0)
    parser.add_argument('--test_ratio', default=0.1)

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=2, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=2, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--save_model', default=False)
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        if arg.phase == 'train':
            self.save_arg()
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        # train_args, test_args = self.arg.train_feeder_args, self.arg.test_feeder_args
        # train_ratio, val_ratio, test_ratio = self.arg.train_ratio, self.arg.val_ratio, self.arg.test_ratio
        # my_feeder = FeederSplit(train_data_m_path=train_args['data_m_path'], train_data_p_path=train_args['data_p_path'], 
        #                                 train_label_path=train_args['label_path'], train_feature_path=train_args['feature_path'],
        #                                 test_data_m_path=test_args['data_m_path'], test_data_p_path=test_args['data_p_path'],
        #                                 test_label_path=test_args['label_path'], test_feature_path=test_args['feature_path'],
        #                                 train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
        # train_set, test_set = my_feeder.get_data()
        # self.data_loader['train'] = torch.utils.data.DataLoader(dataset=train_set, batch_size=self.arg.batch_size, shuffle=True, num_workers=self.arg.num_worker, drop_last=True, worker_init_fn=init_seed)
        # self.data_loader['test'] = torch.utils.data.DataLoader(dataset=test_set, batch_size=self.arg.test_batch_size, shuffle=False, num_workers=self.arg.num_worker, drop_last=False,worker_init_fn=init_seed)

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        # self.loss = nn.BCELoss().cuda(output_device)
        self.loss2=nn.MSELoss().cuda(output_device)
        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        train_class_total_num = np.array([0, 0, 0, 0])
        train_class_true_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0
        for batch_idx, (data_m,data_p, label,feature, index) in enumerate(process):
            if len(label.size()) > 1: train_mode = 'MLL'
            else: train_mode = 'SLL'

            self.global_step += 1
            # get data
            data_m = Variable(data_m.float().cuda(self.output_device), requires_grad=False)
            data_p = Variable(data_p.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            if train_mode == 'MLL': label = label.to(torch.float32)
            feature = Variable(feature.float().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward

            # output, output2 = self.model(data_p,data_m)
            # loss1 = self.loss(output, label)
            # loss2=self.loss2(output2,feature)
            # loss = loss1 + loss2
            output_p,output2,output_m  = self.model(data_p,data_m)
            output=(output_m+output_p)/2

            if train_mode == 'MLL': 
                output_p = F.sigmoid(output_p)
                output_m = F.sigmoid(output_m)

            loss1_m=self.loss(output_m,label)
            loss1_p = self.loss(output_p, label)
            loss2=self.loss2(output2,feature)
            loss =loss1_m+loss1_p +loss2

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            if train_mode == 'SLL':
                value, predict_label = torch.max(output.data, 1)
                total_acc += torch.sum((predict_label == label.data).float())
                cnt += label.size(0)
                trues=list(label.data.cpu().numpy())
                for idx, lb in enumerate(predict_label):
                    train_class_total_num[trues[idx]]+=1
                    train_class_true_num[trues[idx]] += int(lb==trues[idx])
            else:
                total_acc += torch.round(output).eq(label).sum()
                cnt += label.numel()
                class_total_num = torch.round(output).eq(1).sum(axis=0)
                class_true_num = (torch.round(output).eq(label) & label.eq(1)).sum(axis=0)
                for idx in range(len(class_total_num)): train_class_total_num[idx] += class_total_num[idx]
                for idx in range(len(class_true_num)): train_class_true_num[idx] += class_true_num[idx]

            # value, predict_label = torch.max(output.data, 1)
            # acc = torch.mean((predict_label == label.data).float())
            # self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            self.train_writer.add_scalar('loss_1m', loss1_m, self.global_step)
            self.train_writer.add_scalar('loss_1p', loss1_p, self.global_step)
            # self.train_writer.add_scalar('loss_1', loss1, self.global_step)

            self.train_writer.add_scalar('loss_2', loss2, self.global_step)
            # self.train_writer.add_scalar('batch_time', process.iterable.last_duration, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            # if self.global_step % self.arg.log_interval == 0:
            #     self.print_log(
            #         '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
            #             batch_idx, len(loader), loss.data[0], lr))
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))
        
        print('Happy:{},Sad:{},Angry:{},Neutral:{}'.format(train_class_true_num[0]*1.0/train_class_total_num[0],
                                                            train_class_true_num[1]*1.0/train_class_total_num[1],
                                                            train_class_true_num[2]*1.0/train_class_total_num[2],
                                                            train_class_true_num[3]*1.0/train_class_total_num[3]))
        print('Train Accuracy: {: .2f}%'.format(100 * total_acc*1.0 / cnt))

        # if save_model:
        #     state_dict = self.model.state_dict()
        #     weights = OrderedDict([[k.split('module.')[-1],
        #                             v.cpu()] for k, v in state_dict.items()])

            # torch.save(weights, self.arg.model_saved_name + '-' + str(epoch) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        test_class_total_num = np.array([0, 0, 0, 0])
        test_class_true_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0
            # class_right_num=[0,0,0,0]
            # class_total_num=[0,0,0,0]
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data_m,data_p, label,feature, index) in enumerate(process):
                if len(label.size()) > 1: test_mode = 'MLL'
                else: test_mode = 'SLL'
                with torch.no_grad():
                    data_m = Variable(
                        data_m.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    data_p = Variable(
                        data_p.float().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False,
                        volatile=True)
                    if test_mode == 'MLL': label = label.to(torch.float32)
                    output_p, output2,output_m = self.model(data_p,data_m)
                    output=(output_m+output_p)/2
                    if test_mode == 'MLL': output = F.sigmoid(output)
                    # output, output2 = self.model(data_p, data_m)

                    loss = self.loss(output, label)
                    # score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if test_mode == 'SLL':
                        trues=list(label.data.cpu().numpy())
                        for idx, lb in enumerate(predict_label):
                            test_class_total_num[trues[idx]]+=1
                            test_class_true_num[trues[idx]] += int(lb==trues[idx])
                        total_acc += (predict_label == label).sum()
                        cnt += label.size(0)
                    else:
                        total_acc += torch.round(output).eq(label).sum()
                        cnt += label.numel()
                        class_total_num = torch.round(output).eq(1).sum(axis=0)
                        class_true_num = (torch.round(output).eq(label) & label.eq(1)).sum(axis=0)
                        for idx in range(len(class_total_num)): test_class_total_num[idx] += class_total_num[idx]
                        for idx in range(len(class_true_num)): test_class_true_num[idx] += class_true_num[idx]

                    # predicts=list(predict_label.cpu().numpy())
                    # trues=list(label.data.cpu().numpy())
                    # for idx, lb in enumerate(predicts):
                    #     class_total_num[trues[idx]]+=1
                    #     class_right_num[trues[idx]]+=int(lb==trues[idx])

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')
            # score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            accuracy = total_acc * 1.0 / cnt
            # accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
            #     if self.arg.save_model:
            #         state_dict = self.model.state_dict()
            #         weights = OrderedDict([[k.split('module.')[-1],
            #                                 v.cpu()] for k, v in state_dict.items()])
            #         torch.save(weights, self.arg.model_saved_name + '/best_acc.pt')

            # self.lr_scheduler.step(loss)
            # print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            # if self.arg.phase == 'train':
            #     self.val_writer.add_scalar('loss', loss, self.global_step)
            #     self.val_writer.add_scalar('acc', accuracy, self.global_step)

            # score_dict = dict(
            #     zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            # for k in self.arg.show_topk:
            #     self.print_log('\tTop{}: {:.2f}%'.format(
            #         k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
            print('Top1: {:.2f}%'.format(accuracy*100))
            self.print_log('Best acc: {:.2f}%'.format(self.best_acc*100))
            print('Happy:{},Sad:{},Angry:{},Neutral:{}'.format(test_class_true_num[0] * 1.0 / test_class_total_num[0],
                                                                test_class_true_num[1] * 1.0 / test_class_total_num[1],
                                                                test_class_true_num[2] * 1.0 / test_class_total_num[2],
                                                                test_class_true_num[3] * 1.0 / test_class_total_num[3]))
            # if save_score:
            #     with open('{}/epoch{}_{}_score.pkl'.format(
            #             self.arg.work_dir, epoch + 1, ln), 'wb') as f:
            #         pickle.dump(score_dict, f)

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-6:
                    break
                # save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                #         epoch + 1 == self.arg.num_epoch)
                save_model=False
                start = time.time()
                self.train(epoch, save_model=save_model)
                end = time.time()
                print(end - start)

                start = time.time()
                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])
                end = time.time()
                print(end - start)

            # print('best accuracy: ', self.best_acc, ' model_name: ', self.arg.model_saved_name)
            self.print_log('best accuracy: {}'.format(self.best_acc))

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '/wrong.txt'
                rf = self.arg.model_saved_name + '/right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
