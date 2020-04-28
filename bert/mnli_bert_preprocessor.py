import jsonlines
from transformers import BertTokenizer
import numpy as np

# MNLI preprocessor BERT

class BertPreprocessor:
    """
    Class to preprocess ASAG data for bert
    """

    def __init__(self, tokenizer, data=None, data_path=None, save_path=None, mode='test', split=0.95, cls=['[CLS]'],
                 sep=['[SEP]'], pad=['[PAD]'],
                 max_len=128):
        """
        Initialize an instance of BertPreprocessor
        :param mode: Train or test mode
        :param cls: cls token
        :param sep: sep token
        :param pad: pad token
        :param max_len: max length of token sequence
        :param data_path: Path to dataset data
        :param save_path: Path to save preprocessed data
        :param tokenizer: Tokenizer to tokenize the data
        :param data_set_name: name of the data set
        """
        self.max_len = max_len
        self.data_path = data_path
        self.save_path = save_path
        self.tokenizer = tokenizer
        self.cls = cls
        self.sep = sep
        self.pad = pad
        self.mode = mode
        self.split = split
        self.data = data

    def label_to_int(self, lab):
        """
        Assignes label strings to a class
        :param lab: string
        :return: class number
        """
        if lab == 'neutral':
            return 0
        if lab == 'contradiction':
            return 1
        if lab == 'entailment':
            return 2
        else:
            print("whoa")
            raise ValueError

    def token_seg_att(self, seq1, seq2, tokenizer, max_tokens=128):
        """
        Attaches tokens to 2 sentences and transforms tokens to match BERT input
        :param tokenizer: Tokenizer to obtain token id's
        :param seq1: first sentence
        :param seq2: second sentence
        :param cls: CLS token, may be changed
        :param sep: SEP token, may be changed
        :param pad: PAD token, may be changed
        :param max_tokens: max sequence length
        :return: token id's of complete sequence, segmentation mask and attention mask
        """
        tok1 = tokenizer.tokenize(seq1)
        tok2 = tokenizer.tokenize(seq2)
        tokens = self.cls + tok1 + self.sep + tok2
        if len(tokens) > max_tokens - 1:
            tokens = tokens[:max_tokens - 1] + self.sep
        else:
            tokens = tokens + self.sep
        att_len = len(tokens)
        while len(tokens) < max_tokens:
            tokens = tokens + self.pad
        assert (len(tokens) == max_tokens)
        first_sep = tokens.index(self.sep[0])
        tok_ids = np.asarray(tokenizer.convert_tokens_to_ids(tokens))
        seg = np.zeros(max_tokens)
        att = np.append(np.ones(att_len), np.zeros(max_tokens - att_len))
        if att_len < max_tokens:
            seg[first_sep + 1: att_len] = 1
        else:
            seg[first_sep + 1:] = 1
        return tok_ids, seg, att

    def load_data(self):
        """
        Loads data from directory of XML files
        :param path: path to load data from
        :return: list of preprocessed data
        """
        tokenizer = self.tokenizer
        array = []
        # Case if just one case has to be evaluated
        if None != self.data and None == self.data_path:
            data = self.data
            t, s, a = self.token_seg_att(data[0], data[1], tokenizer)
            label = data[2]
            return np.array([t, s, a, label])
        else:
            # MNLI change
            path = self.data_path
            file = jsonlines.open(path)
            for line in file:
                sent1 = line['sentence1']
                sent2 = line['sentence2']
                gold = line['gold_label']
                if gold not in ['entailment', 'contradiction', 'neutral']:
                    continue
                t, s, a = self.token_seg_att(sent1, sent2, tokenizer)
                label = self.label_to_int(gold)
                array.append([t, s, a, label])
            return np.array(array)

    def preprocess(self):
        def create_npy(path, data):
            np.save(path, data, allow_pickle=True)

        NPY = ".npy"
        mode = self.mode
        if mode == 'train':
            data = self.load_data()
            np.random.shuffle(data)
            train, val = np.split(data, [int(len(data) * self.split)])
            train_path = self.save_path + '_train' + NPY
            val_path = self.save_path + '_val' + NPY
            create_npy(train_path, train)
            create_npy(val_path, val)
        elif mode == 'test':
            test_path = self.save_path + '_test' + NPY
            data = self.load_data()
            create_npy(test_path, data)


if __name__ == '__main__':
    # Tokenizer
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True, do_basic_tokenize=True)
    # Train Data
    BertPreprocessor(tokenizer, data_path='../data/datasets/training/mnli/multinli_1.0_train.jsonl',
                     save_path='../data/preprocessed/bert_mnli',
                     mode='train').preprocess()
    # Test Data
    BertPreprocessor(tokenizer, data_path='../data/datasets/testing/mnli/multinli_1.0_dev_matched.jsonl',
                     save_path='../data/preprocessed/bert_mnli_matched',
                     mode='test').preprocess()
    BertPreprocessor(tokenizer, data_path='../data/datasets/testing/mnli/multinli_1.0_dev_mismatched.jsonl',
                     save_path='../data/preprocessed/bert_mnli_mismatched',
                     mode='test').preprocess()

