import numpy as np
import torch.tensor as tensor
from torch.autograd import Variable
import torch


def ndcg_at_k(model, query_label, query_url_feature, k=[5], use_cuda=True):
    cuda = True if torch.cuda.is_available() else False
    FloatTensor = torch.cuda.FloatTensor if cuda and use_cuda else torch.FloatTensor

    ndcg_k = [0.0 for _ in range(len(k))]
    cnt = [0 for _ in range(len(k))]

    for query in query_label.keys():
        # remove train set positive urls
        url_label_dir = query_label[query]
        pred_list = list(set(query_url_feature[query].keys()))
        pred_list_feature = [query_url_feature[query][url] for url in pred_list]
        pred_list_feature = np.asarray(pred_list_feature)
        pred_list_feature = Variable((tensor(pred_list_feature)).type(FloatTensor))

        pred_list_score = model(pred_list_feature).detach().cpu().tolist()

        try:
            pred_url_score = zip(pred_list, pred_list_score)
        except:
            print(query)
            print(pred_list_score)
            return -1.
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)

        for index_k in range(len(k)):
            labels = []
            dcg = 0.0
            for url in url_label_dir:
                labels.append(float(url_label_dir[url]))

            if sum(labels) == 0:
                cnt[index_k] += 1
                continue

            n = len(pred_list) if len(pred_list) < k[index_k] else k[index_k]
            for i in range(0, n):
                (url, score) = pred_url_score[i]
                dcg += (url_label_dir[url] / np.log2(i + 2))  # ((pow(2, url_label_dir[url]) - 1) / np.log2(i + 2))

            # n = k  # len(url_label_dir) if len(url_label_dir) < k else k
            idcg = np.sum(np.array(sorted(labels, reverse=True)[:n]) / np.log2(np.arange(2, n + 2)))

            ndcg_k[index_k] += (dcg / idcg)

            cnt[index_k] += 1

    return [ndcg_k[i] / float(cnt[i]) for i in range(len(k))]
