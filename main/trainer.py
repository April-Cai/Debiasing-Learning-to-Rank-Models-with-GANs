import torch
from torch.autograd import Variable
from torch import tensor
import numpy as np
from model import Dis, Gen
from model import pairwise_loss, estimate_loss
from utils.yahoo_result import output as yahoo_output
# from utils.st_result import output as st_output
from itertools import product
from eval.ndcg import ndcg_at_k
import torch.nn.functional as F


class Trainer:

    def __init__(self, opt, layers):

        # ==== model ====
        self.gen = Gen(n_features=opt.n_features, layers=layers, temperature=opt.temperature)
        self.dis = Dis(n_features=opt.n_features, layers=layers)

        # ==== optimizers
        # Adam
        if opt.opt == 'adam':
            self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=opt.g_lr,
                                                  weight_decay=opt.weight_decay)
            self.optimizer_dis = torch.optim.Adam(self.dis.parameters(), lr=opt.d_lr,
                                                  weight_decay=opt.weight_decay)
        # SGD
        else:
            self.optimizer_gen = torch.optim.SGD(self.gen.parameters(), lr=opt.g_lr,
                                                 weight_decay=opt.weight_decay, momentum=opt.momentum)
            self.optimizer_dis = torch.optim.SGD(self.dis.parameters(), lr=opt.d_lr,
                                                 weight_decay=opt.weight_decay, momentum=opt.momentum)
        # ==== cuda ====

        self.cuda = True if torch.cuda.is_available() and opt.cuda else False
        self.FloatTensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if self.cuda else torch.LongTensor
        if self.cuda:
            self.gen.cuda()
            self.dis.cuda()
        print('use cuda : {}'.format(self.cuda))

        # ==== opt settings ====
        self.n_positions = opt.n_positions
        self.n = opt.n
        self.tools_dir = opt.tools_dir
        self.p = opt.p
        self.norm = opt.norm

        # bias_i, bias_j
        self.t_plus = [1 for _ in range(self.n_positions)]
        self.t_minus = [1 for _ in range(self.n_positions)]
        self.exam = [1 for _ in range(self.n_positions)]

    """
    choose real click docs as pos and GEN output as neg
    """

    def get_sample_pairs(self, features, positions, clicks):

        # == click index ==
        all_f = Variable(features.type(self.FloatTensor))
        clicks = clicks.detach().cpu().numpy()
        index = np.where(clicks == 1)
        unique_group_index, counts_clicks = np.unique(index[0], return_counts=True)

        # == positive and negative features ==
        pos_f = features[index]
        neg_f = []
        pos_p = positions[index]
        neg_p = []

        # == GEN output ==
        count = 0
        g_output = self.gen(all_f).detach().cpu().numpy()
        for i in range(len(unique_group_index)):
            g_index = unique_group_index[i]
            exp_rating = np.exp(g_output[g_index] - np.max(g_output[g_index]))
            # remove clicked docs
            for index in range(counts_clicks[i]):
                exp_rating[pos_p[count + index] - 1] = 0
            count += index + 1
            prob = exp_rating / np.sum(exp_rating, axis=-1)
            try:
                neg_index = np.random.choice(self.n_positions, size=[counts_clicks[i]], p=prob)
            except:
                neg_index = np.random.choice(self.n_positions, size=[counts_clicks[i]])

            choose_index = positions[g_index][neg_index].tolist()

            # invalid samples
            if 0 in choose_index:
                choose_index = positions[g_index][neg_index].tolist()
                neg_index = np.random.choice(self.n_positions, size=[counts_clicks[i]], p=prob)

            neg_f.extend(features[g_index][neg_index].tolist())
            neg_p.extend(choose_index)

        # == output ==
        pos_f = Variable((tensor(pos_f)).type(self.FloatTensor))
        neg_f = Variable((tensor(neg_f)).type(self.FloatTensor))
        pos_p = Variable((tensor(pos_p)).type(self.LongTensor))
        neg_p = Variable((tensor(neg_p)).type(self.LongTensor))

        try:
            pred_valid = self.dis(pos_f).view(-1)
        except:
            print(pos_f.size())
        pred_fake = self.dis(neg_f).view(-1)
        true_diffs = Variable(self.FloatTensor(len(pos_p)).fill_(1), requires_grad=False)

        return pred_valid, pred_fake, true_diffs, pos_p, neg_p

    """
    train the discriminator
    """

    def train_dis(self, pred_valid, pred_fake, true_diffs, position_i, position_j, forward=True):

        pred_diffs = pred_valid - pred_fake

        # calculate pairwise propensity given (i,j)
        propensity = []
        pos_propensity = []
        neg_propensity = []

        for index in range(len(position_i)):
            i = position_i[index]
            j = position_j[index]
            if i != 0 and j != 0:
                prop = self.t_plus[i - 1] * self.t_minus[j - 1]
                if prop != 0:
                    propensity.append(prop)
                else:
                    propensity.append(1)
            else:
                propensity.append(1)
            if i != 0:
                pos_propensity.append(1 / self.t_plus[i - 1])
            else:
                pos_propensity.append(1)

            if j != 0:
                neg_propensity.append(1 / self.t_minus[j - 1])
            else:
                neg_propensity.append(0)

        propensity = Variable((tensor(propensity)).type(self.FloatTensor))
        true_diffs = true_diffs / propensity

        loss = pairwise_loss(pred_diffs, true_diffs)

        # ==== optimize ====
        if forward:
            self.optimizer_dis.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dis.parameters(), self.norm)
            self.optimizer_dis.step()
        return loss.item()

    """
    update t_plus and t_minus
    """

    def estimate_bias(self, pred_valid, pred_fake, true_diffs, position_i, position_j):
        pred_diffs = pred_valid - pred_fake
        pred_diffs = pred_diffs.detach().cpu().numpy()
        true_diffs = true_diffs.detach().cpu().numpy()
        position_i = position_i.detach().cpu().numpy()
        position_j = position_j.detach().cpu().numpy()
        # prepare t_j
        t_j = []
        for index in range(len(position_j)):
            j = position_j[index]
            t_j.append(self.t_minus[j - 1])

        # prepare t_i
        t_i = []
        for index in range(len(position_i)):
            i = position_i[index]
            t_i.append(self.t_plus[i - 1])

        # prepare loss
        loss = []
        for i in range(len(self.t_plus)):
            i = i + 1
            index = np.where(position_i == i)
            pred_diffs_i_j = pred_diffs[index]
            true_diffs_i_j = true_diffs[index]
            t_j_index = np.array(t_j)[index]
            loss_i_j = estimate_loss(pred_diffs=pred_diffs_i_j, true_diffs=true_diffs_i_j) / t_j_index
            loss.append(np.sum(loss_i_j))

        # update t_plus
        if loss[0] == 0:
            loss[0] = np.mean(loss)
        for i in range(len(self.t_plus)):
            if loss[i] == 0:
                loss[i] = np.mean(loss)
            # Eq.(5)
            self.t_plus[i] = np.power(loss[i] / loss[0], 1 / (self.p + 1))

        # prepare loss
        loss = []
        for j in range(len(self.t_minus)):
            j = j + 1
            index = np.where(position_j == j)
            pred_diffs_i_j = pred_diffs[index]
            true_diffs_i_j = true_diffs[index]
            t_i_index = np.array(t_i)[index]
            loss_i_j = estimate_loss(pred_diffs=pred_diffs_i_j, true_diffs=true_diffs_i_j) / t_i_index
            loss.append(np.sum(loss_i_j))

        # update t_minus
        if loss[0] == 0:
            loss[0] = np.mean(loss)
        for i in range(len(self.t_plus)):
            if loss[i] == 0:
                loss[i] = np.mean(loss)
            # Eq.(6)
            self.t_minus[i] = np.power(loss[i] / loss[0], 1 / (self.p + 1))

    """
    main function for training the discriminator and position bias
    """

    def train_dis_prop(self, features, positions, clicks, forward=True, prop=True):

        if torch.sum(clicks) == 0:
            return 0
        # prepare dataset S
        pred_valid, pred_fake, true_diffs, position_i, position_j = self.get_sample_pairs(features, positions, clicks)
        # train dis Eq.(3)
        loss = self.train_dis(pred_valid, pred_fake, true_diffs, position_i, position_j, forward=forward)

        # update position ratios if prop with Eq.(5)(6)
        if prop:
            self.estimate_bias(pred_valid, pred_fake, true_diffs, position_i, position_j)

        return loss

    """
    train the generator
    """

    def train_gen(self, features, forward=True):

        # ==== sample and get reward from DIS====
        features = Variable(features.type(self.FloatTensor))
        d_output = self.dis(features).view(len(features), -1).detach().cpu().numpy()

        # == select n documents for each q ==
        choose_features = []
        choose_reward = []
        for index in range(len(d_output)):
            exp_rating = np.exp(d_output[index] - np.max(d_output[index]))
            prob = exp_rating / np.sum(exp_rating)
            try:
                choose_index = np.random.choice(self.n_positions, size=[self.n], p=prob)
            except:
                choose_index = np.random.choice(self.n_positions, size=[self.n])
            reward = self.dis.reward(features[index][choose_index])
            choose_reward.append(reward.tolist())  # 5 x 10 x 1
            choose_features.append(features[index][choose_index].tolist())

        choose_features = tensor(choose_features).type(self.FloatTensor)
        choose_reward = tensor(choose_reward).type(self.FloatTensor).view(len(choose_features), -1)

        # ==== loss ====
        # update generator using Eq.(7)
        loss = self.gen.score(choose_features, choose_reward)

        # ==== optimize ====
        if forward:
            self.optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.norm)
            self.optimizer_gen.step()
        return loss.item()

    """
    NDCG
    """

    def ndcg(self, l, dis, label, feature, label_i=5):
        label_index = l.index(label_i)
        if dis:
            res = ndcg_at_k(self.dis, label, feature, k=l, use_cuda=self.cuda)
            label = res[label_index]
            for i in range(len(l)):
                print('ndcg@{}:{:.4f} '.format(l[i], res[i]))
        else:
            res = ndcg_at_k(self.gen, label, feature, k=l, use_cuda=self.cuda)
            label = res[label_index]
            for i in range(len(l)):
                print('ndcg@{}:{:.4f} '.format(l[i], res[i]))
        return label

    """
    evaluate
    """

    def evaluate(self, model_path, since, opt, data_dir, data):
        print('')
        print('==== eval {} ===='.format(model_path))

        if model_path == "":
            print('You have to specify the eval model')
        else:
            self.dis.load_state_dict(torch.load(model_path))
            yahoo_output(model=self.dis, data=data, data_dir=data_dir,
                         tools_dir=opt.tools_dir,
                         opt=opt, model_path=model_path, since=since, use_cuda=self.cuda)
