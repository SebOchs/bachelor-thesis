import bert.utils as utils
from transformers import BertTokenizer

# Script to preprocess data for training the BERT model

# scientsBank
# Data paths
TRAIN_ROOT = '../data/training/sciEntsBank'
TEST_UA_ROOT = '../data/testing/sciEntsBank/test-unseen-answers'
TEST_UD_ROOT = '../data/testing/sciEntsBank/test-unseen-domains'
TEST_UQ_ROOT = '../data/testing/sciEntsBank/test-unseen-questions'

# Data files
TRAIN_FILE = '../data/train.npy'
VAL_FILE = '../data/val.npy'
TEST_UA_FILE = '../data/test_ua.npy'
TEST_UD_FILE = '../data/test_ud.npy'
TEST_UQ_FILE = '../data/test_uq.npy'

# beetle
BEETLE_ROOT = '../data/training/beetle'
BEETLE_FILE = '../data/beetle_training.npy'

# Tokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

# Other constants
MAX_LENGTH = 128
CLS = ['[CLS]']
SEP = ['[SEP]']
PAD = ['[PAD]']

# scientsBank
train_data = utils.load_data(TRAIN_ROOT, tokenizer)
utils.create_npy(train_data, [TRAIN_FILE, VAL_FILE], mode='train')
ua_data = utils.load_data(TEST_UA_ROOT, tokenizer)
utils.create_npy(ua_data, TEST_UA_FILE)
ud_data = utils.load_data(TEST_UD_ROOT, tokenizer)
utils.create_npy(ud_data, TEST_UD_FILE)
uq_data = utils.load_data(TEST_UQ_ROOT, tokenizer)
utils.create_npy(uq_data, TEST_UQ_FILE)

#beetle
beetle_data = utils.load_data(BEETLE_ROOT, tokenizer)
utils.create_npy(beetle_data, BEETLE_FILE)
