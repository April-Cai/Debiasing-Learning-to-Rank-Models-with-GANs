import numpy as np
import os
import torch.tensor as tensor
from torch.autograd import Variable
import torch
import datetime


def output(model, data, data_dir, tools_dir, model_path, opt, since, use_cuda):
    qid_list_map = {}
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    for query in data.session_query_url_label_test.keys():
        pred_list = data.session_query_url[query]  # all doc id given query
        pred_list_feature = [data.session_query_url_feature[query][url] for url in
                             data.session_query_url[query]]  # all features given (query,doc)

        pred_list_feature = np.asarray(pred_list_feature)
        pred_list_feature = Variable(tensor(pred_list_feature).type(FloatTensor))

        pred_list_score = model(pred_list_feature).view(-1).detach().cpu().tolist()

        pred_url_score = zip(pred_list, pred_list_score)
        pred_url_score = sorted(pred_url_score, key=lambda x: x[1], reverse=True)  # rerank_lists
        qid_list_map[query] = pred_url_score

    with open(data_dir + '/test/test.ranklist', 'w') as file:
        for q in qid_list_map:
            for i in range(len(qid_list_map[q])):
                query = q.split('_')[1]
                file.write(
                    '{} Q0 test_{}_{} {} {} RankLSTM\n'.format(query, query, str(qid_list_map[q][i][0]), str(i + 1)
                                                               , str(qid_list_map[q][i][1])))
    with open('../log/{}.txt'.format(opt.log), 'a+') as file:
        print()
        file.write('[start] {}\n[end  ] {}\n[model] {}\n'.format(since, datetime.datetime.now(), model_path))
        file.write('[param] {}\n'.format(opt))
        # ndcg@1,3,5,10
        os.system('{}/trec_eval -c -m ndcg_cut.1,3,5,10 {}/test/test.qrels {}/test/test.ranklist > res.out'.
                  format(tools_dir, data_dir, data_dir))
        res = ''.join(open('./res.out').readlines())
        print(res)
        file.write(res)

        # map
        os.system('{}/trec_eval -c -m map {}/test/test.qrels {}/test/test.ranklist > res.out'.
                  format(tools_dir, data_dir, data_dir))
        res = ''.join(open('./res.out').readlines())
        print(res)
        file.write(res)

        # info
        os.system('{}/trec_eval -c {}/test/test.qrels {}/test/test.ranklist > res.out'.format(tools_dir, data_dir,
                                                                                              data_dir))

        res = ''.join(open('./res.out').readlines())
        print(res)
        file.write(res)
