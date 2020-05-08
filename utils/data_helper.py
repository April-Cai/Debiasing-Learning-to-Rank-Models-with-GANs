from utils import click_models as cm
import json
import random
from torch.utils.data import Dataset, DataLoader
from torch import tensor


# ========================= DataReader =============================

class DataReader:

    def __init__(self, feature_file, qrel_file, json_file, feature_size=700, use_real_label=False, cut=-1):
        self.feature_file = feature_file
        self.qrel_file = qrel_file
        self.feature_size = feature_size
        self.json_file = json_file
        if not use_real_label:
            self.model = self.ck_model()
        self.use_real_label = use_real_label
        self.cut = cut  # cut means group cut for Tiangong-ST dataset

        self.session_query_url_feature = {}  # {query_key: {url:[feature]}}
        self.session_query_url = {}  # {query_key:[url]}  # save position inherited
        self.get_feature()  # session_query_url_feature, session_query_url

        self.session_query_url_label_train = {}  # {query_key:{url, label}}
        self.session_query_url_label_valid = {}
        self.session_query_url_label_test = {}
        self.get_label()  # session_query_url_label_train, session_query_url_label_valid, session_query_url_label_test

        self.print_info()

    """
    click model
    """

    def ck_model(self):
        with open(self.json_file) as fin:
            model_desc = json.load(fin)
            return cm.loadModelFromJson(model_desc)

    """
    Get all query,url,index,feature
    session_query_url_feature : {query_key:{url:[feature]}}
    session_query_url : {query_key:[url]}

    """

    def get_feature(self):
        current_query = ''
        session_index = 0

        # .feature file: test_{query}_{url} 0:0.74142 6:0.78235 ... 699:0.74142
        for root in [self.feature_file, self.feature_file.replace('train', 'valid'),
                     self.feature_file.replace('train', 'test')]:

            with open(root) as fin:
                count = 0
                for line in fin:
                    count += 1
                    cols = line.strip().replace('"', '').split()
                    query = str(cols[0].split('_')[1])
                    url = str(cols[0].split('_')[2])

                    feature = [0.0 for _ in range(self.feature_size)]
                    for x in cols[1:]:
                        arr = x.split(':')
                        feature[int(arr[0])] = float(arr[1])

                    # update session
                    if query != current_query:
                        session_index += 1
                        current_query = query
                        key = str(session_index) + '_' + current_query
                        self.session_query_url_feature[key] = {url: feature}
                        self.session_query_url[key] = [url]

                    else:
                        key = str(session_index) + '_' + current_query
                        if url not in self.session_query_url_feature[key]:  # avoid repeating
                            self.session_query_url_feature[key][url] = feature
                            self.session_query_url[key].append(url)

    """
    read into label dict {query_key:{url:label}}
    query_url_label_train
    query_url_label_valid
    query_url_label_test
    
    """

    def get_label(self):
        # .qrels file: {query} 0 test_{query}_{url} {label}
        qrel_root = [self.qrel_file, self.qrel_file.replace('train', 'valid'),
                     self.qrel_file.replace('train', 'test')]
        qrel_dict = [self.session_query_url_label_train, self.session_query_url_label_valid,
                     self.session_query_url_label_test]

        current_query = ''
        session_index = 0

        for i in range(3):
            with open(qrel_root[i]) as fin:
                count = 0
                for line in fin:
                    count += 1
                    items = line.strip().split()
                    if items.__len__() == 4:
                        col = items[2].split('_')
                        label = int(items[3])
                    else:
                        col = items[0].split('_')
                        label = int(items[1])

                    query = str(col[-2])
                    url = str(col[-1])

                    if query != current_query:
                        session_index += 1
                        current_query = query
                    key = str(session_index) + '_' + current_query
                    if key not in qrel_dict[i]:
                        qrel_dict[i][key] = {url: label}
                    else:
                        qrel_dict[i][key][url] = label

    """
    sample part or all of queries(groups) from training set
    return [query_key]
    """

    def sample_train_positive(self, num_samples=-1):
        all_query = list(self.query_url_label_train.keys())
        num_query = len(all_query)
        query_list = []
        # part of keys no repeat
        if num_samples != -1:
            for _ in range(num_samples):
                index = random.randint(0, num_query - 1)
                while all_query[index] in query_list:
                    index = random.randint(0, num_query - 1)
                query_list.append(all_query[index])
        # all the keys
        else:
            query_list = list(self.query_url_label_train.keys())

        return query_list

    """
    print dataset information
    """

    def print_info(self):
        print('Load train {} valid {} test {}'.format(len(self.session_query_url_label_train),
                                                      len(self.session_query_url_label_valid),
                                                      len(self.session_query_url_label_test)))
        print('all query {}'.format(len(self.session_query_url_feature)))

    """
    sample clicks with click model and human labels
    """

    def sample_clicks(self, labels):
        clicks = []
        if self.model:
            clicks, _, _ = self.model.sampleClicksForOneList(labels)
            while sum(clicks) == 0:
                clicks, _, _ = self.model.sampleClicksForOneList(labels)
        return clicks


# ========================= RankDataset =============================

# dataset
# query_list: n
# feature_list: n x n_positions x n_features
# click_list: n x n_positions
# pos_list: n x n_positions
class RankDataset(Dataset):

    def __init__(self, feature_list, click_list, pos_list):
        self.feature_list = feature_list
        self.click_list = click_list
        self.pos_list = pos_list

    def __getitem__(self, index):
        return self.feature_list[index], self.click_list[index], self.pos_list[index]

    def __len__(self):
        return self.feature_list.size(0)


# ========================= prepare dataset for DataLoader =============================

class RankDataHelper:
    def __init__(self, flags):

        self.flags = flags
        self.data_dir, self.cut = self.load_data()

        self.data = DataReader(feature_file=self.data_dir + '/train/train.feature',
                               qrel_file=self.data_dir + '/train/train.qrels',
                               json_file=self.data_dir + '/setting/{}'.format(flags.cm),
                               feature_size=flags.n_features,
                               use_real_label=flags.use_label, cut=self.cut)

    def load_data(self):
        if self.flags.data == 'Y':
            return self.flags.y_dir, -1
        elif self.flags.data == 'S':
            return self.flags.s_dir, -1

    def load_into_tensor(self, group='train', num_samples=-1):

        feature_list = []
        click_list = []
        position_list = []
        label_list = []

        if group == 'train':
            query_url_label = self.data.session_query_url_label_train
        elif group == 'test':
            query_url_label = self.data.session_query_url_label_test
        elif group == 'valid':
            query_url_label = self.data.session_query_url_label_valid
        else:
            assert 'Invalid group'
            return None

        if num_samples == -1:
            keys = query_url_label.keys()

        else:
            keys = self.data.sample_train_positive(num_samples=num_samples)

        for query in keys:
            features = []
            labels = []
            positions = []

            for i in range(0, min(len(self.data.session_query_url[query]), self.flags.n_positions)):
                url = self.data.session_query_url[query][i]
                features.append(self.data.session_query_url_feature[query][url])
                try:
                    labels.append(query_url_label[query][url])
                except:
                    print(query_url_label[query], self.data.session_query_url[query])
                positions.append(i + 1)

            # sample clicks
            if not self.flags.use_label:
                clicks = self.data.sample_clicks(labels)
                while len(clicks) < self.flags.n_positions:
                    clicks.append(0)
                click_list.append(clicks)

            # fit size
            while len(labels) < self.flags.n_positions:
                positions.append(0)
                features.append([0 for _ in range(self.flags.n_features)])
                labels.append(0)

            feature_list.append(features)
            position_list.append(positions)
            label_list.append(labels)

        # choose to use human labels or user clicks
        if self.flags.use_label:
            return tensor(feature_list), tensor(label_list), tensor(position_list)
        else:
            return tensor(feature_list), tensor(click_list), tensor(position_list)

    def get_data_loader(self, group, num_samples=-1):
        feature, click, position = self.load_into_tensor(group, num_samples)
        dataset = RankDataset(feature_list=feature, click_list=click, pos_list=position)
        return DataLoader(dataset=dataset, batch_size=self.flags.batch_size,
                          num_workers=self.flags.n_cpu, shuffle=True)
