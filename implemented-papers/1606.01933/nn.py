import numpy as np

import torch
import torch.nn as nn
from torchtext.legacy import datasets
from torchtext.legacy import data
from torchtext import vocab

'''
FOLDER_PATH - folder should include the GloVe embedding and SNLI datasets
'''
FOLDER_PATH = "./gdrive/My Drive/Master/SharingWork/1606.01933/data/snli_1.0/"


'''
definition used to convert variables so it can accelarted by cuda
'''
deviceCuda = torch.device("cuda")
deviceCPU = torch.device("cpu")
USE_CUDA = True
RUNNING_LOCAL = False
if RUNNING_LOCAL:
    FOLDER_PATH = './data/snli_1.0/'
    USE_CUDA = False

'''
GloVe embeddings
'''
GLOVE_VECTORS = vocab.GloVe(name='840B', dim=300, cache=FOLDER_PATH+'glove_cache/')

'''
Models stops training when the accuracy is not different than the average over
last AVG_ACC_STEPS steps
'''
AVG_ACC_STEPS = 5

'''
DB - holds and handles datasets

defines how the datasets are read
- Use spacy to tokenize the data, and return the length of each of the examples
- Reduce the embedding vocabulary to reduce memory use.
'''
class DB(object):
    def __init__(self, batch_size):
        self.data_field = data.Field(init_token='NULL', tokenize='spacy',
                                     batch_first=True, include_lengths=True)
        self.label_field = data.Field(sequential=False, batch_first=True)

        self.label_field.build_vocab([['contradiction'], ['entailment'],
                                     ['neutral']])

        self.train_ds, self.dev_ds, self.test_ds = datasets.SNLI.splits(self.data_field, self.label_field, root=FOLDER_PATH)

        # Build initial vocabulary using all the words in GLOVE
        self.data_field.build_vocab(self.train_ds, self.dev_ds, self.test_ds,
                                    vectors=GLOVE_VECTORS,
                                    unk_init=torch.Tensor.normal_)

        # Build a new vocabulary that include only the words from GLOVE that are
        # in train-dev-test
        from collections import Counter

        fake_keys = Counter(list(self.data_field.vocab.stoi.keys()))
        self.glove_keys = [[key] for key in GLOVE_VECTORS.stoi.keys() if fake_keys[key] > 0]
        self.data_field.build_vocab(self.glove_keys, vectors=GLOVE_VECTORS)
        fake_keys = []

        # Use BucketIterator to batch together examples with similar length
        # in order to reduce padding needed
        self.train_iter, self.dev_iter, self.test_iter =\
        data.BucketIterator.splits((self.train_ds, self.dev_ds, self.test_ds),
                                   batch_size=batch_size, device=deviceCPU,
                                   sort_key=lambda d: len(d.premise),
                                   shuffle=True, sort=True)

    def getIter(self, iter_type):
        if iter_type == "train":
            return self.train_iter
        elif iter_type == "dev":
            return self.dev_iter
        elif iter_type == "test":
            return self.test_iter
        else:
            raise Exception("Invalid type")

class Tagger(nn.Module):
    '''
    init() -
    @embedding_dim - embedding dimensions
    @projected_dim - dimension to project vectors to
    @tagset_size - number of classes
    @vectors - embedding vectors
    @f_dim, v_dim - dimension of inner FFN layers

    The vectors doesn't include embeddings for unknown words and for padding,
    these are added here manually, where they are drawn from normal distribution

    Unkown words are represented by index 0, padding words are represented by
    index 1, define embedding value 0 for padding so it will have no effect
    '''
    def __init__(self, embedding_dim, projected_dim, tagset_size,
                 vectors, f_dim=200, v_dim=200):
        super(Tagger, self).__init__()
        self.embedding_dim = embedding_dim
        # Embedding unknown words is done randomly to one of n_unk_vecs vecs
        self.n_unk_vecs = 100

        # Create Embeddings for unkown words
        vecs = vectors/torch.norm(vectors, dim = 1, keepdim = True)
        self.unknown_idx = vectors.shape[0]
        pad = torch.randn((self.n_unk_vecs, vecs[0].shape[0]))
        vecs = torch.cat((vecs, pad), 0)
        vecs[0] = torch.zeros(vecs[0].shape)
        vecs[1] = torch.zeros(vecs[0].shape)
        # Model layers
        self.wembeddings = nn.Embedding.from_pretrained(embeddings=vecs, freeze=True)
        self.project = nn.Linear(embedding_dim, projected_dim)
        self.G = self.feedForward(f_dim * 2, v_dim, 0.2)
        self.H = self.feedForward(v_dim * 2, v_dim, 0.2)
        self.linear = nn.Linear(v_dim, tagset_size)
        self.hidden_dim = projected_dim
        self.f_dim = f_dim
        self.F = self.feedForward(self.hidden_dim, f_dim, 0.2)
        self.softmax = nn.Softmax(dim=1)

    '''
    feedForward() - FF layer with optional dropout
    @i_dim - input dimension
    @o_dim - output dimension
    @dropout - if dropout > 0 then it will be used
    '''
    def feedForward(self, i_dim, o_dim, dropout):
        use_dropout = dropout > 0
        layers = []

        layers.append(nn.Linear(i_dim, o_dim))
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

        layers.append(nn.Linear(o_dim, o_dim))
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.ReLU())

        layers = nn.Sequential(*layers)
        return layers

    '''
    forawrd() - one pass of tagger model
    @premise_data - premise sentence
    @hyp_data - hypothesis sentence

    uknown words are indexed by random indexes pointing to the dedicated
    vectors for unkown words.

    Create a mask to remove the effect of padding words
    '''
    def forward(self, premise_data, hyp_data):
        prem_rand_idx = torch.randint(self.unknown_idx,
                                      self.unknown_idx + self.n_unk_vecs -1,
                                      premise_data.shape)
        hyp_rand_idx = torch.randint(self.unknown_idx,
                                     self.unknown_idx + self.n_unk_vecs -1,
                                     hyp_data.shape)

        premise_data[premise_data == 0] = prem_rand_idx[premise_data == 0]
        hyp_data[hyp_data == 0] = hyp_rand_idx[hyp_data == 0]

        premise_mask = torch.ones(premise_data.shape)
        hyp_mask = torch.ones(hyp_data.shape)
        premise_mask[premise_data == 1] = 0
        hyp_mask[hyp_data == 1] = 0

        tmp = list(premise_mask.shape)
        tmp.append(1)
        premise_mask = premise_mask.reshape(tmp).repeat(1,1,200)

        tmp = list(hyp_mask.shape)
        tmp.append(1)
        hyp_mask = hyp_mask.reshape(tmp).repeat(1,1,200)


        if USE_CUDA:
            padded_premise_w = premise_data.to(deviceCuda)
            padded_hyp_w = hyp_data.to(deviceCuda)
            premise_mask = premise_mask.to(deviceCuda)
            hyp_mask = hyp_mask.to(deviceCuda)
        else:
            padded_premise_w = premise_data
            padded_hyp_w = hyp_data


        prem_w_e = self.wembeddings(padded_premise_w)
        hyp_w_e = self.wembeddings(padded_hyp_w)

        # Project the embeddings to smaller vector
        prem_w_e = self.project(prem_w_e)
        hyp_w_e = self.project(hyp_w_e)

        a = prem_w_e
        b = hyp_w_e
        fa = self.F(a)
        fb = self.F(b)

        E = torch.bmm(fa, torch.transpose(fb, 1, 2))

        # Calculate softmax once for every query
        E4beta = self.softmax(E.view(-1, b.shape[1]))
        E4beta = E4beta.view(E.shape)
        beta = torch.bmm(E4beta, b)

        E4alpha = torch.transpose(E, 1, 2)
        saved_shape = E4alpha.shape
        E4alpha = self.softmax(E4alpha.reshape(-1, a.shape[1]))
        E4alpha = E4alpha.view(saved_shape)
        alpha = torch.bmm(E4alpha, a)

        # Attend
        ##Concat to each it's weights
        weighted_a = torch.cat((prem_w_e, beta), 2)
        weighted_b = torch.cat((hyp_w_e, alpha), 2)

        ##Masked Feedforward
        v1 = self.G(weighted_a)*premise_mask
        v2 = self.G(weighted_b)*hyp_mask

        # Aggregate
        v1 = torch.sum(v1, 1)
        v2 = torch.sum(v2, 1)

        h_in = torch.cat((v1, v2), 1)
        y = self.H(h_in)
        y = self.linear(y)

        if USE_CUDA:
          y = y.to(deviceCPU)

        return y

    def getLabel(self, data):
        _, prediction_argmax = torch.max(data, 1)
        return prediction_argmax

'''
Run() - Auxilary class used to help train, run, and measure Tagger

Class makes it easier to check different configurations for the Tagger. Each
configuration can be set in the params data structure and then we can run
multiple configurations of Tagger easier.

'''
class Run(object):
    def __init__(self, params):
        self.edim = params['EMBEDDING_DIM']
        self.rnn_h_dim = params['RNN_H_DIM']
        self.num_epochs = params['EPOCHS']
        self.batch_size = params['BATCH_SIZE']
        self.train_file = params['TRAIN_FILE']
        self.dev_file = params['DEV_FILE']
        self.run_dev = params['RUN_DEV']
        self.learning_rate = params['LEARNING_RATE']
        self.acc_data_list = []
        self.acc_test_list = []
        self.load_params = params['LOAD_PARAMS']

    def _calc_batch_acc(self, tagger, flatten_tag, flatten_label):
        predicted_tags = tagger.getLabel(flatten_tag)
        diff = predicted_tags - flatten_label
        correct_cntr = len(diff[diff == 0])  # tmp
        total_cntr = len(predicted_tags)  # - to_ignore
        return correct_cntr, total_cntr

    def _flat_vecs(self, batch_tag_score, batch_label_list):
        flatten_tag = batch_tag_score  # .reshape(-1, batch_tag_score.shape[2])
        flatten_label = torch.LongTensor(batch_label_list)  # .reshape(-1))
        return flatten_tag, flatten_label

    '''
    runOnDev() - Runs the tagger on validation/test set
    tagger - Tagger instance
    data_iter - iterator on relevant data
    acc_list - accuracy list
    d_type: "Validation" or "Test"
    '''
    def runOnDev(self, tagger, data_iter, acc_list, d_type):
        tagger.eval()

        with torch.no_grad():
            correct_cntr = 0
            total_cntr = 0
            data_iter.init_epoch()
            for sample in data_iter:
                premise_data, _ = sample.premise
                hyp_data, _ = sample.hypothesis
                batch_label = (sample.label - torch.ones(sample.label.shape)).long()

                batch_tag_score = tagger.forward(premise_data, hyp_data)

                # calc accuracy
                batch_label_tensor = torch.LongTensor(batch_label)
                c, t = self._calc_batch_acc(tagger, batch_tag_score, batch_label_tensor)
                correct_cntr += c
                total_cntr += t

        acc = correct_cntr / total_cntr
        acc_list.append(acc)
        print(d_type + " accuracy " + str(acc))

        tagger.train()
    '''
    train() - train over training dataset, optionally testing on validation set
    '''
    def train(self):

        print("Loading data")
        db = DB(self.batch_size);
        train_iter = db.getIter("train")
        print("Done loading data")

        print("init tagger")
        tagger = Tagger(embedding_dim=self.edim, projected_dim=self.rnn_h_dim,
                        tagset_size=3, vectors = db.data_field.vocab.vectors)

        print("done")

        if USE_CUDA:
          tagger.to(deviceCuda)

        print("define loss and optimizer")
        loss_function = nn.CrossEntropyLoss()  # ignore_index=len(lTran.tag_dict))
        optimizer = torch.optim.Adagrad(tagger.parameters(), lr=self.learning_rate,
                                        initial_accumulator_value=0.1, weight_decay=0.000002)  # 0.01)
        print("done")

        if self.run_dev:
            self.runOnDev(tagger, db.getIter('dev'),
                          self.acc_data_list, "Validation")
            self.runOnDev(tagger, db.getIter('test'),
                          self.acc_test_list, "Testset")
        for epoch in range(self.num_epochs):
            train_iter.init_epoch()
            loss_acc = 0
            correct_cntr = 0
            total_cntr = 0
            for sample in train_iter:
                tagger.zero_grad()

                premise_data, _ = sample.premise
                hyp_data, _ = sample.hypothesis
                batch_label = (sample.label - torch.ones(sample.label.shape)).long()

                batch_tag_score = tagger.forward(premise_data, hyp_data)
                # flatten_tag, flatten_label = self._flat_vecs(batch_tag_score, batch_label_list)

                # calc accuracy
                batch_label_tensor = torch.LongTensor(batch_label)
                c, t = self._calc_batch_acc(tagger, batch_tag_score, batch_label_tensor)
                correct_cntr += c
                total_cntr += t

                loss = loss_function(batch_tag_score, batch_label_tensor)
                loss_acc += loss.item()
                loss.backward()
                optimizer.step()

                tagger.zero_grad()

            if self.run_dev:
                self.runOnDev(tagger, db.getIter('dev'),
                              self.acc_data_list, "Validation")

            print("epoch: " + str(epoch) + " " + str(loss_acc))
            self.train_accuracy = correct_cntr/total_cntr
            self.train_loss = loss_acc
            print("Train accuracy " + str(correct_cntr/total_cntr))

            if len(self.acc_data_list) < AVG_ACC_STEPS:
                avg_acc = 0
            else:
                avg_acc = np.sum(self.acc_data_list[-AVG_ACC_STEPS:])/AVG_ACC_STEPS
            if np.abs(self.acc_data_list[-1] - avg_acc) < 0.001:
                break

        if self.run_dev:
            self.runOnDev(tagger, db.getIter('dev'),
                          self.acc_data_list, "Validation")
            self.runOnDev(tagger, db.getIter('test'),
                          self.acc_test_list, "Testset")


'''
FAVORITE_RUN_PARAMS - Parameters we saw that words best with the Tagger on the
SNLI dataset.
'''
FAVORITE_RUN_PARAMS = {
    'EMBEDDING_DIM': 300,
    'RNN_H_DIM': 200,
    'BATCH_SIZE': 32,
    'LEARNING_RATE': 0.05
}

if __name__ == "__main__":
    train_file = FOLDER_PATH + "snli_1.0_train.jsonl"
    epochs = 200
    run_dev = True
    dev_file = FOLDER_PATH + "snli_1.0_dev.jsonl"

    RUN_PARAMS = FAVORITE_RUN_PARAMS
    RUN_PARAMS.update({
                'TRAIN_FILE': train_file,
                'DEV_FILE' : dev_file,
                'RUN_DEV' : run_dev,
                'EPOCHS' : epochs,
                'LOAD_PARAMS': False,
                })

    run = Run(RUN_PARAMS)
    run.train()
