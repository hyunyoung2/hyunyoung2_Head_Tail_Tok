"""
With head-tail corpus generated using https://github.com/hyunyoung2/Hyunyoung_HEAD_TAIL_CORPUS

with BUFS-JBNUCorpus2020 from https://github.com/bufsnlp2030/BUFS-JBNUCorpus2020

This code splits data into train and test. 

the structure of head-tail corpus is as follows:

# text = 잠실에서 우리가 더 강했다 .
# head_tail_text = 잠실+에서 우리+가 더 강했+다 .

# text = 이 정부가 , 국민에게 탄핵당한 정부가 왜 이렇게 사드 배치를 서두르는지 이해할 수가 없다 .
# head_tail_text = 이 정부+가 , 국민+에게 탄핵당한 정부+가 왜 이렇+게 사드 배치+를 서두르+는지 이해할 수+가 없+다 .

For head-tail corpus, 

The sentence starting with '# text =' is the original sentence and then


the sentence starting with '# head_tail_text =' is to split each word on the original text into 

a pari of head-tail token on the original sentence

The following is the special text to detect whether text is original or head-tail sentence

## train and test corpus 
LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string

For train and test, 

There are three pairs of train and test, which are kcc_q28_only, kcc_150_only, kcc_150_n_q28

train corpus is named 'train.txt'

test corpus is named 'text.txt'

"""
#-*- coding: utf-8 -*-

from glob import glob
import os
from collections import OrderedDict


ROOT_DIR = "corpus/"

DICT_ROOT = "dict/"

CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

# For Debugging option, increasing level enumerate log in detail
DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

## if DEBUG_LEVEL is 0, No print for debugging
DEBUG = DEBUG_LEVEL[0]

## train and test corpus 
LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string

TRAIN_CORPUS = "train.txt"
TEST_CORPUS = "test.txt"

START_TOK = "<START_TOK>"
END_TOK = "<END_TOK>"

## Dictionary type 
DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", 
              "RIGHT_BI", "RIGHT_TRI",
              "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

DICT_HEAD =  ["N-gram_word","Heat-tail_Token","Fre.","Prob."]

def read_raw_corpus(path):
    """Reading data line by line from head-tail copurs from corpus directory

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    the corpus format is the following:

    # text = 잠실에서 우리가 더 강했다 .
    # head_tail_text = 잠실+에서 우리+가 더 강했+다 .

    # text = 이 정부가 , 국민에게 탄핵당한 정부가 왜 이렇게 사드 배치를 서두르는지 이해할 수가 없다 .
    # head_tail_text = 이 정부+가 , 국민+에게 탄핵당한 정부+가 왜 이렇+게 사드 배치+를 서두르+는지 이해할 수+가 없+다 .

    Arg:
      path(str): The path to a raw corpus to be read 
                 from corpus/#_kcc_000/train.txt or corpus/#_kcc_000/text.txt
    return:
      data(list): A list of lines in raw corpus like 
                  ["line1_str", "line2_str", ..., "line_n_str"]
    """

    assert isinstance(path, str), "The type of input is wrong 'read_raw_corpus' function: the type of path is {}, the value is {}".format(type(path), path)


    with open(path, "r") as wr:
        data = [val.strip() for val in wr.readlines()] # to strip the front and back from the text

    
    # check duplication and details
    sent_copy = OrderedDict()
    hetail_copy = OrderedDict()
    for idx, val in enumerate(data):
        ## train and test corpus 
        ## LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string
        if LINE_OPTS[0] in val:
            sents = val
           
            assert len(val.split()) > 3, "your data is empty, {}".format(val)
            # duplication check
            if sent_copy.get(val):
                assert False, "original sentence has duplication in dir {}, {}".format(path, val)
            else:
                sent_copy[val] = 1

        elif LINE_OPTS[1] in val:
            hetail = val # head_tail_texts

            assert len(val.split()) > 3, "your data is empty, {}".format(val)
            # duplication check
            if hetail_copy.get(val):
                assert False, "head_tail_text has duplication in dir {}, {}".format(path, val)
            else:
                hetail_copy[val] = 1

        elif LINE_OPTS[2] == val: ## in order to chek whether FROM and HEAD-TAIL is the same or not,  compare pair of FROM and HEAD-TAIL
            temp_sent = sents.split()
            temp_hetail = hetail.split()
            for temp_sent_idx, temp_sent_val in enumerate(temp_sent):
                if temp_sent_idx < 3:
                    continue
                else:
                    assert temp_sent_val == "".join(temp_hetail[temp_sent_idx].split('+')), "In the 'read_raw_corpus' function, tem_sent_val != temp_hetail[temp_sent_idx], val('{}') and temp_hetail['{}']('{}')".format(temp_sent_val, temp_sent_idx, temp_hetail[temp_sent_idx])

        else:  
            assert False, "read_raw_corpus function error"
      

    if DEBUG in DEBUG_LEVEL[1:]:
        print("\n===== Reading a file of {} =====".format(path))
        print("The number of lines: {}".format(len(data)))
        print("for top 5 of examples, \n{}".format(data[0:5]))

    return data


def count_tok_and_line_num(file_data):
    """Extract ngram from original sentence

    ngram and head_tail token are coupled as a pair. 

    with corpus/#_kcc_000/train.txt or corpus/#_kcc_000/text.txt

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    when creating bi and tri gram, START_TOK and END_TOK is used.

    the corpus format is the following:

    # text = 잠실에서 우리가 더 강했다 .
    # head_tail_text = 잠실+에서 우리+가 더 강했+다 .

    # text = 이 정부가 , 국민에게 탄핵당한 정부가 왜 이렇게 사드 배치를 서두르는지 이해할 수가 없다 .
    # head_tail_text = 이 정부+가 , 국민+에게 탄핵당한 정부+가 왜 이렇+게 사드 배치+를 서두르+는지 이해할 수+가 없+다 .

    Args:
       file_data(list): sentences read corpus line by line 

    Return:
       None
    """

    # The number of lines
    lines_num = 0

    # The number of tokens
    tokens_num = 0

    # sent dict 
    sents_dict = OrderedDict()
 
    ## LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string
    for idx, val in enumerate(file_data):
        if LINE_OPTS[0] in val: # original sentence
            sent = val.split()[3:]

            lines_num += 1
            tokens_num += len(sent)

            _key = " ".join(sent)
            if sents_dict.get(_key):
                sents_dict[_key] += 1
            else:
                sents_dict[_key] = 1
        
        
    if DEBUG in DEBUG_LEVEL[0:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check count_tok_and_line_num function =====")
        print("The total of sents: {}".format(lines_num))
        print("The total of tokens: {}".format(tokens_num))
        print("The top examplses: \n{}".format(sorted(sents_dict.items(), key=lambda x: x[1], reverse=True)[0:5]))

    return sents_dict 

def chekc_data_set(file_path):
    """Check the dataset 

    using the structure of directory 
    
    ROOT_DIR = "corpus/"

    DICT_ROOT = "dict/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    TRAIN_CORPUS = "train.txt"
    TEST_CORPUS = "test.txt"

    Args: 
       file_path(str): the location of train or test dataset
    Return: 
       sents_dict(dict): a pair of sentence and its number 
    """

    list_of_files = glob(file_path+"*", recursive=True)

    files_sorted = sorted(list_of_files)

    assert len(files_sorted) == 1, "making dictionary is wrong!!, files_sorted: {}".format(files_sorted)

    if DEBUG in DEBUG_LEVEL[1:]:
        print("\n===== a list of files to be read =====")
        print("The destination of dic: {}".format(file_path))
        print("The type of 'list_of_files': {}".format(type(list_of_files)))
        print("The number of files: {}".format(len(list_of_files)))
        print("For examples, \n{}".format(list_of_files))
        print("\n===== a list of files is sorted =====")
        print("The type of 'files_sorted': {}".format(type(files_sorted)))
        print("The number of files sorted: {}".format(len(files_sorted)))
        print("For examples sorted: {}".format(files_sorted))
 

    for idx, val in enumerate(files_sorted):
        print("\n===== Preprocessing the file {} under dicrectory {} =====".format(val, file_path))

        raw_data = read_raw_corpus(val)

        print("\n===== Reading the file {} is done ! =====".format(val))

        print("\n===== Counting lines and tokens starts =====")

        sents_duplication = count_tok_and_line_num(raw_data)

        print("\n===== Counting lines and tokens is done ! =====") 

        assert sorted(sents_duplication.items(), key=lambda x: x[1], reverse=True)[0][1] == 1, "you text has the dupilcation"

    return sents_duplication

def check_train_has_test(train_data, test_data):
    """Check train data also has data which is in test data
    
    Args:
        train_data(dict): train sentence data, (sents, num)
        test_data(idct): test sentence data, (sents, num)

    Return:
        None
    """

    for _key, _val in test_data.items():
    
        if train_data.get(_val):
            print("\n========= Detect it =======")
            print("The data in train data: {}, {}".format(_key, train_data(_key)))
            print("The data in test data: {}, {}".format(_key, _val))
            input()
    
    print("\n====== Checking 'check_train_has_test' function is done ! =====") 

 
if __name__ == "__main__":
    """
    ROOT_DIR = "corpus"

    DICT_ROOT = "dict/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    # For Debugging option, increasing level enumerate log in detail
    DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

    ## if DEBUG_LEVEL is 0, No print for debugging
    DEBUG = DEBUG_LEVEL[0]

    ## train and test corpus 
    LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string

    TRAIN_CORPUS = "train.txt"
    TEST_CORPUS = "test.txt"

    START_TOK = "<START_TOK>"
    END_TOK = "<END_TOK>"
 

    ## dictionary type 
    ##                 0  ,     1    ,      2    ,      3    ,      4     ,              5             ,                 6        
    ## DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]
    """

    child_dict_type = CHILD_DIR[2]

    ## For Test
    train_dataset_loc = ROOT_DIR+TRAIN_CORPUS #child_dict_type+TRAIN_CORPUS

    test_dataset_loc = ROOT_DIR+TEST_CORPUS # child_dict_type+TEST_CORPUS

    ## For Experiments
    ##train_dataset_loc = ROOT_DIR+child_dict_type+TRAIN_CORPUS

    ##test_dataset_loc = ROOT_DIR+child_dict_type+TEST_CORPUS

    train_sents_dict = chekc_data_set(train_dataset_loc)

    test_sents_dict = chekc_data_set(test_dataset_loc)

 
    check_train_has_test(train_sents_dict, test_sents_dict)

