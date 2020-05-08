import argparse
import random
import numpy as np
import torch
import config
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_helper import RankDataHelper
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer

if __name__ == "__main__":

    # ======================================= args =================================================
    parser = argparse.ArgumentParser()
    # ==== data ====
    parser.add_argument("--data", type=str, default=config.data, help="data type: S(ample) or Y(ahoo)")
    parser.add_argument("--y_dir", type=str, default=config.y_dir, help="yahoo directory")
    parser.add_argument("--s_dir", type=str, default=config.s_dir, help="sample directory")
    parser.add_argument("--tools_dir", type=str, default=config.tools_dir, help="eval tool directory")
    parser.add_argument("--cm", type=str, default='pbm_0.1_1_4_1.json', help="click model json file")
    parser.add_argument("--n_positions", type=int, default=config.n_positions, help="number of pad positions")
    parser.add_argument("--n_features", type=int, default=700, help="size of each image dimension")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=config.n_cpu, help="number of cpu threads for data loader")
    # ==== training ====
    parser.add_argument("--seed", type=int, default=config.seed, help="random seed")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--d_lr", type=float, default=config.d_lr, help="discriminator learning rate")
    parser.add_argument("--g_lr", type=float, default=config.g_lr, help="generator learning rate")

    parser.add_argument("--n", type=int, default=5, help="sample numbers for generator")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="l2 regularization")
    parser.add_argument("--mode", type=int, default=1, help="train or eval")
    parser.add_argument("--model", type=str, default='point', help="eval model")
    parser.add_argument('--des', type=str, default='', help='description')
    parser.add_argument("--use_label", type=lambda x: (str(x).lower() == 'true'), default='False',
                        help="use qrels or not")

    parser.add_argument("--dis", type=int, default=config.d_epochs, help="discriminator rel epoch")
    parser.add_argument("--gen", type=int, default=config.g_epochs, help="generator rel epoch")
    parser.add_argument('--opt', type=str, default='sgd', help='optimizer')
    parser.add_argument('--norm', type=float, default=5.0, help='clip norm')
    parser.add_argument('--log', type=str, default='log', help='log file')
    parser.add_argument('--momentum', type=float, default=0.0, help='momentum')
    parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
    parser.add_argument("--cuda", type=lambda x: (str(x).lower() == 'true'), default='True', help="use cuda")
    parser.add_argument("--p", type=float, default=0.05, help="regularization")
    parser.add_argument("--prop_frequency", type=int, default=1, help="propensity estimation frequency")
    parser.add_argument("--layers", type=list, default=[512, 256, 128], help="network structure")

    opt = parser.parse_args()
    print(opt)

    # ======================================= random seed =================================================

    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    # ======================================= data loader =================================================

    data_helper = RankDataHelper(flags=opt)
    train_loader = data_helper.get_data_loader('train')
    train_iterator = iter(train_loader)
    feature = data_helper.data.session_query_url_feature
    valid = data_helper.data.session_query_url_label_valid
    data_dir = data_helper.data_dir


    def get_batch(iterator):
        try:
            f, c, p = next(iterator)
        except:
            iterator = iter(data_helper.get_data_loader('train'))
            f, c, p = next(iterator)
        if torch.sum(c) == 0:
            return get_batch(iterator)
        else:
            return f, c, p, iterator


    # ======================================= model and log =================================================

    model = Trainer(opt, opt.layers)
    print(model.dis)
    print(model.gen)

    writer = SummaryWriter()
    with open('../log/{}.txt'.format(opt.log), 'a+') as file:
        import datetime

        since = datetime.datetime.now()
        file.write('{}\n'.format(datetime.datetime.now()))
        file.write('{}'.format(model.dis))
        file.write('{}'.format(model.gen))

    # ======================================= train and eval =================================================
    # training mode
    if opt.mode == 1:

        best_dis_model = ''
        best_dis_loss = 0.0
        dis = 1
        gen = 1
        loop = 0
        best_dis_ndcg_5 = 0.0

        while loop < opt.epochs:

            # ==================================== D ====================================
            loop += 1
            print('=' * 20)
            print('Epoch {}'.format(loop))

            # ============== DIS ============

            print('\n== Discriminator ==')
            dis_loss = 0.0

            # d-step: train the discriminator and update position bias (if d-step mod e-step = 0)
            for i in range(opt.dis):
                f, c, p, train_iterator = get_batch(train_iterator)

                if i % opt.prop_frequency == 0:
                    loss = model.train_dis_prop(features=f, clicks=c, positions=p, prop=True)
                else:
                    loss = model.train_dis_prop(features=f, clicks=c, positions=p, prop=False)

                dis_loss += loss
                if i % 10 == 0:
                    print('loop {} {:.4f}'.format(i + 1, loss))

            dis_loss = dis_loss / (i + 1)
            print('avg loss {:.4f}'.format(dis_loss))
            writer.add_scalar('loss/dis', dis_loss, dis)
            dis += 1

            # output position bias
            print('Position ratios')
            print('{:4} {:6s} {:6s}'.format('', 't_plus', 't_minus'))
            for index in range(len(model.t_plus)):
                print('{:4} {:6.2f} {:6.2f}'.format(index + 1, model.t_plus[index], model.t_minus[index]))

            # ndcg@5 on validation set
            ndcg_5 = model.ndcg([5], True, valid, feature, label_i=5)
            writer.add_scalar('dis/ndcg@5/eval', ndcg_5, loop)
            # save best model for best_dis_ndcg_5
            if ndcg_5 > best_dis_ndcg_5:
                best_dis_ndcg_5 = ndcg_5
                best_dis_model = '/model/best_dis_ndcg_{:.4f}.pth'.format(best_dis_ndcg_5)
                save_dir = data_dir + best_dis_model
                torch.save(model.dis.state_dict(), save_dir)
                print(' saved to {}'.format(save_dir))

            # ============== GEN ============
            if opt.gen > 0:
                print('\n== Generator ==')
                gen_loss = 0.0

                # g-step: train generator
                for i in range(opt.gen):
                    f, c, p, train_iterator = get_batch(train_iterator)
                    loss = model.train_gen(features=f)
                    gen_loss += loss
                    if i % 10 == 0:
                        print('loop {} {:.4f}'.format(i + 1, loss))

                gen_loss = gen_loss / (i + 1)
                print('loss {:.4f}'.format(loop, gen_loss))
                writer.add_scalar('loss/gen', gen_loss, gen)
                gen += 1

                # ndcg@5 on validation set
                ndcg_5 = model.ndcg([5], False, valid, feature, label_i=5)
                writer.add_scalar('gen/ndcg@5/eval', ndcg_5, loop)

            # shows performance on test set after training
            print('\ntime:', datetime.datetime.now() - since, '\n')
            if loop == opt.epochs:
                with open('../log/{}.txt'.format(opt.log), 'a+') as file:
                    file.write('\nEpoch {} \n'.format(loop))
                    for i in range(len(model.t_plus)):
                        file.write('{:4} {:6.2f} {:6.2f}\n'.format(i + 1, model.t_plus[i], model.t_minus[i]))
                model.evaluate(model_path=data_dir + best_dis_model, opt=opt, data=data_helper.data,
                               data_dir=data_dir,
                               since=since)

    # evaluation mode
    elif opt.mode == 0:
        model.evaluate(model_path=data_dir + '/model/' + opt.model, opt=opt, data=data_helper.data, data_dir=data_dir,
                       since=since)
