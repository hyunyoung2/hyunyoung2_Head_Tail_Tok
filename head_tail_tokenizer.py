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

TEST_RESULT_ROOT = "test_result/"

CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

# For Debugging option, increasing level enumerate log in detail
DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

## if DEBUG_LEVEL is 0, No print for debugging
DEBUG = DEBUG_LEVEL[0]

## train and test corpus 
LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string
TEST_LINE_OPT = "# output = "

TRAIN_CORPUS = "train.txt"
TEST_CORPUS = "test.txt"

START_TOK = "<START_TOK>"
END_TOK = "<END_TOK>"

## Dictionary type 
##             0   ,     1    ,      2    ,     3     ,      4     ,               5            ,                 6
DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

DICT_HEAD =  ["N-gram_word","Heat-tail_Token","Fre.","Prob."]


def read_test_raw_corpus(path):
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

    assert isinstance(path, str), "The type of input is wrong in 'read_raw_corpus' function: the type of path is {}, the value is {}".format(type(path), path)


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
      

    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== Reading a file of {} =====".format(path))
        print("The number of lines: {}".format(len(data)))
        print("for top 5 of examples, \n{}".format(data[0:5]))

    return data

def read_n_gram_dict(path, dict_idx="None"):
    """Reading data line by line from head-tail dict

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    the corpus format is the following:

    N-gram_word\tHeat-tail_Token\tFre.\tProb.\n

    잠실에서\t잠실+에서\t3\t60.0
    잠실에서\t잠실에서\t2\t40.0
    우리가\t우리+가\t4\t100.0


    there are 7 types of dictionaries as follows: 
 
    ## Dictionary type 

    ##              0  ,     1    ,      2    ,     3     ,      4     ,             5              ,                 6
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

    Arg:
      path(str): The path to a raw corpus to be read 
                 from dict/0000_dict
    return:
      data(list): A list of lines with the format above like 
                  ["잠실에서\t잠실+에서\t3\t60.0",
                   "잠실에서\t잠실에서\t2\t40.0",
                   "우리가\t우리+가\t4\t100.0", ...]
      pair_dict(dict): pair dictionary
                       {n-gram_0: [(head_tail_token_1, cnt), (head_tail_token_2, cnt), ..., 
                        n_gram_1: [(head_tail_token_1, cnt), (head_tail_token_2, cnt)]
 
    """

    assert isinstance(path, str), "The type of input is wrong 'read_n_gram_dict' function: the type of path is {}, the value is {}".format(type(path), path)

    assert dict_idx in DICT_TYPES, "There 7 types of DICT_TYPES, the current dict_type: {}".format(dict_idx)

    with open(path, "r") as wr:
        data = [val.strip().split("\t") for val in wr.readlines()] # to strip the front and back from the text

    
    # check whether the number of columns is 4
    # DICT_HEAD =  ["N-gram_word","Heat-tail_Token","Fre.","Prob."]

    hetail_dict = OrderedDict()
    for idx, val in enumerate(data):
        if idx == 0:  # head line 
            continue

        assert len(val) == 4, "Your data is wrong, {}('{}')".format(val, len(val))

        if dict_idx == DICT_TYPES[0]:
            _key = val[0]  ## N-gram word -> tuple change later on

            ## check this for debugging
            ##print("_key: {}".format(_key))
            ##input()
 
        else:
            _key = tuple(val[0].split())

            ## Check dict for degugging
            ##print("_key: {}".format(_key))
            ##input()


            if dict_idx == DICT_TYPES[0]: # UNI
                assert isinstance(_key, str), "DICT_TYPE is wrong, _key: {}".format(_key)
            elif dict_idx in [DICT_TYPES[1], DICT_TYPES[3]]: # "LEFT_BI", "RIGHT_BI"
                assert len(_key) == 2, "You _key is wrong on dict_type {}, _key: {}".format(dict_idx, _key)
            elif dict_idx in [DICT_TYPES[2], DICT_TYPES[4]]: # "LEFT_TRI", "RIGHT_TRI"
                assert len(_key) == 3, "You _key is wrong on dict_type {}, _key: {}".format(dict_idx, _key)
            elif dict_idx == DICT_TYPES[5]: # "BIDIRECTIONAL_BI_WINDOW_1"
                assert len(_key) == 3, "You _key is wrong on dict_type {}, _key: {}".format(dict_idx, _key)
            elif dict_idx == DICT_TYPES[6]: # "BIDIRECTIONAL_TRI_WINDOW_1"
                assert len(_key) == 5, "You _key is wrong on dict_type {}, _key: {}".format(dict_idx, _key)

        _value = val[1] ## Head-tail tok
        _fre = int(val[2]) ## Fre.
        _prob = float(val[3]) ## Prob.

        if hetail_dict.get(_key):
            hetail_dict[_key].append((_value, _fre, _prob))
        else:
            hetail_dict[_key] = []
            hetail_dict[_key].append((_value, _fre, _prob))

    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== Reading a file of {} =====".format(path))
        print("The number of lines: {}".format(len(data)))
        print("The head: {}".format(data[0]))
        print("for top 5 of examples, \n{}".format(data[1:5]))
        print("The total pair_dict: {}".format(len(hetail_dict)))
        print("The top 2 examples with list(hetail_dict.items()): {}".format(list(hetail_dict.items())[0:5]))
 
    return data[0], data[1:], hetail_dict

def extract_ngram(data, dictionary_type="None"):
    """Extract ngram from original sentence

    when creating bi and tri gram, START_TOK and END_TOK is used.

    the corpus format is the following:

    # text = 잠실에서 우리가 더 강했다 .
    # head_tail_text = 잠실+에서 우리+가 더 강했+다 .

    # text = 이 정부가 , 국민에게 탄핵당한 정부가 왜 이렇게 사드 배치를 서두르는지 이해할 수가 없다 .
    # head_tail_text = 이 정부+가 , 국민+에게 탄핵당한 정부+가 왜 이렇+게 사드 배치+를 서두르+는지 이해할 수+가 없+다 .


    this function uses '# text =' line as input

    the input is string type and then this function change string into list with split function 

    Args:
       file_data(list): sentences read corpus line by line 

    Returns:
       uni_word(list): pairs of (unigram FROM, HEAD-TAIL-TOKEN)

       left_bi_word(list): pairs of (left bigram FROM, HEAD-TAIL-TOKEN)
       left_tri_word(list): pairs of (left trigram FROM, HEAd-TAIL-TOKEN)
    
       right_bi_word: pairs of (right bigram FROM, HEAD-TAIL-TOKEN) 
       right_tri_word: pairs of (right trigram FROM, HEAD-TAIL-TOKEN)

       bidir_word_win_1: pairs of (left and right bigram FROM, HEAD-TAIL-TOKEN) # left 1 and right 1
       bidir_word_win_2: pairs of (left and right trigram FROM, HEAD-TAIL-TOKEN) # left 2 and right 2
    """
 
    assert isinstance(data, list), "The type of input is wrong 'extract_ngram' function: the type of data is {}, the value of data is {}".format(type(data), data)

    assert dictionary_type in DICT_TYPES, "The type of dictionary type is wrong 'extract_ngram' function: the dictionary type is {}".format(dictionary_type)
    
    sent = data # data is the same form from sent
    
    # unigram word
    uni_word = []

    # for bidirectional bi-gram and tri-gram 
    # this left
    left_bi_word = []
    left_tri_word = []
    
    # for bidirectional bi-gram and tri-gram
    # this right 
    right_bi_word = []
    right_tri_word = []

    ## for joint probability with bidirectional 
    ## Later on  
    bidir_word_win_1 = [] # left 1 and right 1
    bidir_word_win_2 = [] # left 2 and right 2
    
    len_sent = len(sent)
    for temp_sent_idx, temp_sent_val in enumerate(sent):
 
        ##DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", 
        ##      "RIGHT_BI", "RIGHT_TRI",
        ##      "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]
        ## If you want to utilize the two or more dictionary type, 
        ### change the if elif else statement below
  
        if dictionary_type == DICT_TYPES[0]:
            # uni_word
            uni = (temp_sent_val)
            uni_word.append(uni)
         
        elif dictionary_type == DICT_TYPES[1]:
                
            ## left section
            # left bi 
            if temp_sent_idx == 0:
                left_bi = (START_TOK, temp_sent_val)
                left_bi_word.append(left_bi)
            else: # temp_sent_idx != 0:
                left_bi = (sent[temp_sent_idx-1], temp_sent_val)
                left_bi_word.append(left_bi)
          
        elif dictionary_type == DICT_TYPES[2]:    
            # left tri 
            if temp_sent_idx == 0:
                left_tri = (START_TOK, START_TOK, temp_sent_val)
                left_tri_word.append(left_tri)
            elif temp_sent_idx == 1:
                left_tri = (START_TOK, sent[temp_sent_idx-1], temp_sent_val)
                left_tri_word.append(left_tri)
            else: # 1 < temp_sent_idx and  temp_sent_idx < len(sent):
                left_tri = (sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val)
                left_tri_word.append(left_tri)
                 
        elif dictionary_type == DICT_TYPES[3]:
            # right section
            reverse_sent_idx = (temp_sent_idx+1) * -1

            # right bi 
            if reverse_sent_idx == -1:
                right_bi = (sent[reverse_sent_idx], END_TOK)
                right_bi_word.append(right_bi)
            elif len_sent * -1 <= reverse_sent_idx and  reverse_sent_idx < -1:
                right_bi = (sent[reverse_sent_idx], sent[reverse_sent_idx+1])
                right_bi_word.append(right_bi)
            else:
                assert False, "Reverse_sent_idx < len(sent) * -1 or Reverse_sent_idx > -1 for right bi"

        elif dictionary_type == DICT_TYPES[4]:
            # right section
            reverse_sent_idx = (temp_sent_idx+1) * -1

            # right tri
            if reverse_sent_idx == -1:
                right_tri = (sent[reverse_sent_idx], END_TOK, END_TOK)
                right_tri_word.append(right_tri)
            elif reverse_sent_idx == -2:
                right_tri = (sent[reverse_sent_idx], sent[reverse_sent_idx+1], END_TOK)
                right_tri_word.append(right_tri)
            elif len_sent * -1 <= reverse_sent_idx and reverse_sent_idx < -2:
                right_tri = (sent[reverse_sent_idx], sent[reverse_sent_idx+1], sent[reverse_sent_idx+2])
                right_tri_word.append(right_tri)
            else:
                assert False, "Reverse_sent_idx < len(sent) * -1 or Reverse_sent_idx > -1 for right tri"

        elif dictionary_type == DICT_TYPES[5]:

            # bidirectional section
            # bidirectional word with window 1 
            if len_sent == 1:
                bi_word_window_1 = (START_TOK, temp_sent_val, END_TOK)
                bidir_word_win_1.append(bi_word_window_1)
            else:
                if temp_sent_idx == 0:
                    bi_word_window_1 = (START_TOK, temp_sent_val, sent[temp_sent_idx+1])
                    bidir_word_win_1.append(bi_word_window_1)
                elif 0 < temp_sent_idx and temp_sent_idx < len_sent - 1:
                    bi_word_window_1 = (sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1])
                    bidir_word_win_1.append(bi_word_window_1)
                elif temp_sent_idx == len_sent -1: 
                    bi_word_window_1 = (sent[temp_sent_idx-1], temp_sent_val, END_TOK)
                    bidir_word_win_1.append(bi_word_window_1)
                else:
                    assert False, "temp_sent_idx < 0, for bidirectional word with window 1"

        elif dictionary_type == DICT_TYPES[6]:

      
            # bidirectional word with window 2 
            if len_sent == 1:
                bi_word_window_2 = (START_TOK, START_TOK, temp_sent_val, END_TOK, END_TOK)
                bidir_word_win_2.append(bi_word_window_2) 
            elif len_sent == 2:
                if temp_sent_idx == 0:
                    bi_word_window_2 = (START_TOK, START_TOK, temp_sent_val, sent[temp_sent_idx+1], END_TOK)
                    bidir_word_win_2.append(bi_word_window_2)
                elif temp_sent_idx == 1:
                    bi_word_window_2 = (START_TOK, sent[temp_sent_idx-1], temp_sent_val, END_TOK, END_TOK)
                    bidir_word_win_2.append(bi_word_window_2)
            elif len_sent == 3:
                if temp_sent_idx == 0:
                    bi_word_window_2 = (START_TOK, START_TOK, temp_sent_val, sent[temp_sent_idx+1], sent[temp_sent_idx+2])
                    bidir_word_win_2.append(bi_word_window_2)
                elif temp_sent_idx == 1:
                    bi_word_window_2 = (START_TOK, sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1], END_TOK)
                    bidir_word_win_2.append(bi_word_window_2)
                elif temp_sent_idx == 2:
                    bi_word_window_2 = (sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val, END_TOK, END_TOK)
                    bidir_word_win_2.append(bi_word_window_2)    
            else: # len_sent > 3
                if temp_sent_idx == 0:
                    bi_word_window_2 = (START_TOK, START_TOK, temp_sent_val, sent[temp_sent_idx+1], sent[temp_sent_idx+2])
                    bidir_word_win_2.append(bi_word_window_2)
                elif temp_sent_idx == 1:
                    bi_word_window_2 = (START_TOK, sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1], sent[temp_sent_idx+2])
                    bidir_word_win_2.append(bi_word_window_2)
                elif 1 < temp_sent_idx and temp_sent_idx < len_sent - 2:
                    bi_word_window_2 = (sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val , sent[temp_sent_idx+1], sent[temp_sent_idx+2])
                    bidir_word_win_2.append(bi_word_window_2)
                elif temp_sent_idx == len_sent -2: 
                    bi_word_window_2 = (sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1], END_TOK)
                    bidir_word_win_2.append(bi_word_window_2)
                elif temp_sent_idx == len_sent -1:
                    bi_word_window_2 = (sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val, END_TOK, END_TOK)
                    bidir_word_win_2.append(bi_word_window_2)
                else:
                    assert False, "temp_sent_idx < 0, for bidirectional word with window 2"

        else: 
             assert False, "When you call 'extract ngram' function, it is wrong dictionary type is {}".format(dictionary_type)

    ## DEBUGGING 
    ##print("\nSent (len, elements): {}\n{}".format(len(sent), sent))
    ##print("\n===== Uni-gram word =====")
    ##print("uni_word (len, elements): {}\n{}".format(len(uni_word), uni_word))
    ##print("\n===== For letf n-gram word =====")
    ##print("\nleft_bi_word (len, elements): {}\n{}".format(len(left_bi_word), left_bi_word))
    ##print("\nleft_tri_word (len, elements): {}\n{}".format(len(left_tri_word), left_tri_word))
    ##print("\n===== For right n-gram word =====")
    ##print("\nright_bi_word (len, elements): {}\n{}".format(len(right_bi_word), right_bi_word))
    ##print("\nright_tri_word (len, elements): {}\n{}".format(len(right_tri_word), right_tri_word))
    ##print("\n===== For bidirectinal n-gram word =====")
    ##print("\nbidir_word_win_1 (len, elements): {}\n{}".format(len(bidir_word_win_1), bidir_word_win_1))
    ##print("\nbidir_word_win_2 (len, elements): {}\n{}".format(len(bidir_word_win_2), bidir_word_win_2))
    ##input()
 
        
    if DEBUG in DEBUG_LEVEL[1:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check extract ngram function =====")
        print("The sent: {}, {}".format(len(sent), sent))
        print("\n===== Uni-gram word =====")
        print("The number of uni_word: {}".format(len(uni_word)))
        print("The top 5 examples of uni_word: {}".format(uni_word[0:5]))
        print("\n===== For left n-gram word =====")
        print("The number of left_bi_word: {}".format(len(left_bi_word)))
        print("The top 5 examples of left_bi_word: {}".format(left_bi_word[0:5]))
        print("\nThe number of left_tri_word: {}".format(len(left_tri_word)))
        print("The top 5 examples of left_tri_word: {}".format(left_tri_word[0:5]))
        print("\n===== For right n-gram word =====")
        print("The number of right_bi_word: {}".format(len(right_bi_word)))
        print("The top 5 examples of right_bi_word: {}".format(right_bi_word[0:5]))
        print("\nThe number of right_tri_word: {}".format(len(right_tri_word)))
        print("The top 5 examples of right_tri_word: {}".format(right_tri_word[0:5]))
        print("\n===== For bidirectinal n-gram word =====")
        print("The number of bidir_word_win_1: {}".format(len(bidir_word_win_1)))
        print("The top 5 examples of bidir_word_win_1: {}".format(bidir_word_win_1[0:5]))
        print("\nThe number of bidir_word_win_2: {}".format(len(bidir_word_win_2)))
        print("The top 5 examples of bidir_word_win_2: {}".format(bidir_word_win_2[0:5]))
        
        
    return uni_word, left_bi_word, left_tri_word, right_bi_word, right_tri_word, bidir_word_win_1, bidir_word_win_2


def match_with_dict(input_sent, ngram_data, ngram_dict_data, _ground_truth):
    """matching n_gram_word with n_gram_dict 

    and then extract head-tail token pairs

    this is dependent on DICT_TYPES:

    ## Dictionary type 
    ##             0   ,     1    ,      2    ,     3     ,      4     ,               5            ,                 6
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]


    dict is pair_dict which consists of the following:

               {n-gram_0: [(head_tail_token_1, cnt), (head_tail_token_2, cnt), ..., 
                n_gram_1: [(head_tail_token_1, cnt), (head_tail_token_2, cnt)]

    Args:
       input_sent(listt): input sentences like [tok1, tok2, ..., tokn]
       ngram_data(list): pairs of ngram with a sentence
       ngram_dict_data(dict): paired dict sorted by 'cnt' like
                             {n-gram_0: [(head_tail_token_1, cnt), (head_tail_token_2, cnt), ..., 
                              n_gram_1: [(head_tail_token_1, cnt), (head_tail_token_2, cnt)]

       _ground_truth(list): list of ground_truth head_tail-toks

    Returns:
       answer(list): a sequence of head-tail tokens
       original_word_num(int): the nubmer of words not in dictionary
       matching_num(int): the original sentence[idx]'s tok == ground_truth[idx]'s tok, additionally count 1 
    """

    answer = []
    original_word_num = 0
    matching_num = 0     
     

    original_sent = input_sent

    assert len(original_sent) == len(ngram_data), "len(original_set) != len(ngram_data), original_sent('{}'): {} and ngram_data('{}'): {}".format(len(original_sent), original_sent, len(ngram_data), ngram_data)

    for idx, val in enumerate(ngram_data):
         if ngram_dict_data.get(val): ## Keep in mind dictionary is sorted 
             answer.append(ngram_dict_data[val][0][0])
             ## For Debugging
             ##print("The orginal: {} , ngram_dict: {}".format(original_sent[idx], ngram_dict_data[val])) 
         else:
             original_word_num += 1
             if _ground_truth[idx] == original_sent[idx]:
                 ##print(_ground_truth[idx], original_sent[idx])
                 ##input()
                 matching_num += 1
                 
                 # For else case 
                 ##print(_ground_truth[idx], original_sent[idx])
                 ##input()
       
             answer.append(original_sent[idx])
             ## For debugging
             ##print("The orginal: {} , ngram_dict: None".format(original_sent[idx]))

         ## Check this For Debugging
         ##print("val, original_sent[idx], answer, original_word_num")
         ##print(val, original_sent[idx], answer, original_word_num)
         ##input()

    if DEBUG in DEBUG_LEVEL[1:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check 'match_with_dict' function =====")
        print("The len of original_sent: {}\n{}".format(len(original_sent), original_sent))
        print("The len of ngram_data: {}\n{}".format(len(ngram_data), ngram_data))
 

    return answer, original_word_num, matching_num


def test(test_file_path, dict_file_path, test_result_file_path, dict_type="None"):
    """execute head-tail tokenization with dictrionary for head_tail tokens on test dataset 

    using the structure of directory 
    
    ROOT_DIR = "corpus/"

    DICT_ROOT = "dict/"

    TEST_RESULT_ROOT = "test_result/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    TRAIN_CORPUS = "train.txt"
    TEST_CORPUS = "test.txt"

    ## Dictionary type 
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

   
    After counting n-gram From and head-tail token, write dictionary into DICT_ROOT+CHILD_DIR

    Args: 
       test_file_path(str): the location of test dataset
       dict_file_path(str): the location of dictionary 
       tes_result_file_path(str): the location where result stores
       dict_type(str): the type of dictionary of head-tail token
                       one of the followings

                       DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]
    Return: 
       None 
    """

    assert dict_type in DICT_TYPES, "your dict_type in 'test' function is wrong, dict_type: {}".format(dict_type)

    files_sorted = [test_file_path] # If handling various files 

    assert len(files_sorted) == 1, "your length of files_sorted is wrong, the length of files_sorted is {}".format(files_sorted)


    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== a list of files to be read =====")
        print("The destination of dic: {}".format(test_file_path))
        print("The type of 'files_sorted': {}".format(type(files_sorted)))
        print("The number of files sorted: {}".format(len(files_sorted)))
        print("For top 3 of examples sorted: {}".format(files_sorted))
     
    for _idx, _val in enumerate(files_sorted):
        print("\n===== Preprocessing the file {} =====".format(_val))

        raw_data = read_test_raw_corpus(_val)

        print("\n===== Reading the file {} is done ! =====".format(_val))

    _, _, _dict = read_n_gram_dict(path=dict_file_path, dict_idx=dict_type)


    assert isinstance(test_result_file_path, str), "The test_result_file_path is wrong in 'test' function: the type of path is {}, the value is {}".format(type(test_result_file_path), test_result_file_path)


    ## Test result file open 
    test_result_file_writer = open(test_result_file_path, "w")

    print("\n======= writing test result file into {} is open ! ===========".format(test_result_file_path))

    ## train and test corpus 
    ## LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string
    ## TEST_LINE_OPT = "# output = "

    ### TEST file generation
    ## First, creating test file 
    ## second, measure accuracy 
    total_tok_num = 0
    matched_tok_num = 0
    non_matched_tok_num_with_dict = 0
    original_word_num_matched_without_dict = 0

    for idx, val in enumerate(raw_data):
        if LINE_OPTS[0] in val:

            ##print("\nInput: {}".format(" ".join(val.split()[3:])))
            test_result_file_writer.write(val+"\n")

            temp_sent = val.split()[3:]

            ##print("Input sentence: {}\n{}".format(len(temp_sent), temp_sent))

        elif LINE_OPTS[1] in val:

            ##print("Solution: {}".format(" ".join(val.split()[3:])))
            test_result_file_writer.write(val+"\n")

            ground_truth_toks = val.split()[3:]

            ##print("Ground_truth: {}\n{}".format(len(ground_truth_toks), ground_truth_toks))
 
        elif LINE_OPTS[2] == val:           
            uni, left_bi, left_tri, right_bi, right_tri, bidir_bi_win_1, bidir_tri_win_2 =  extract_ngram(temp_sent, dict_type)

            ##                 0  ,     1    ,     2     ,      3    ,      4     ,              5             ,              6
            ## DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

            if dict_type == DICT_TYPES[0]: # UNI
                ngram_pair = uni
            elif dict_type == DICT_TYPES[1]: # LEFT_BI
                ngram_pair = left_bi
            elif dict_type == DICT_TYPES[2]: # LEFT_TRI
                ngram_pair = left_tri
            elif dict_type == DICT_TYPES[3]: # RIGHT_BI
                ngram_pair = right_bi[::-1] ## reverse
            elif dict_type == DICT_TYPES[4]: # RIGHT_TRI
                ngram_pair = right_tri[::-1] ## reverese
            elif dict_type == DICT_TYPES[5]: # BIDIRECTIONAL_BI_WINDOW_1
                ngram_pair = bidir_bi_win_1 
            elif dict_type == DICT_TYPES[6]: # BIDIRECTIONAL_TRI_WINDOW_2
                ngram_pair = bidir_tri_win_2
            else: 
                assert False, "Your dict types is wrong in matching, dict type is {}".format(dict_type) 
            
                 
            result, original_word_num, original_word_num_matched = match_with_dict(temp_sent, ngram_pair, _dict, ground_truth_toks)

            ## Check if the length of ground_truth and output of our model is the same or not
            assert len(result) == len(temp_sent), "check if len(result) == len(temp_sent) or not, result('{}'){} and temp_sent('{}'){}".format(len(result), result, len(temp_sent), temp_sent)


            ## counting function !!!! here ####
            total_tok_num += len(temp_sent)
           
            ### Considering dictionary 
            non_matched_tok_num_with_dict += original_word_num
            original_word_num_matched_without_dict += original_word_num_matched
            ## matching or not matching, just count matching cases 
            ## Debugging
            ##print("Ground_Truth: {}".format(ground_truth_toks))
            ##print("Matched: {}".format(result))

            for predicted_idx, predicted_val in enumerate(ground_truth_toks):
                if predicted_val == result[predicted_idx]:
                    matched_tok_num += 1
                    ## For Debugging
                    ##print("The sum: {}, The predicted: {}, ground_truth_toks: {}".format(matched_tok_num, result[predicted_idx], predicted_val))
                ## For Debugging        
                ##print("The sum: {}, The predicted: {}, ground_truth_toks: {}".format(matched_tok_num, predicted_val, ground_truth_toks[predicted_idx]))
                ##input()
            ## For Debugging
            ##print("mathec_tok_num: {}".format(matched_tok_num))
            ##input()

            ##print("OUTPUT: {}".format(" ".join(result)))
            test_result_file_writer.write(TEST_LINE_OPT+" ".join(result)+"\n\n")

            ##print("The original_word_num: {}".format(original_word_num))
 
        else:
            assert False, "Your dict type is wrong, the type of dict is {}".format(val)

    #### The result is printed in here #####
    print("\n======= Test function is done ! ===========")
    test_result_file_writer.close()
    print("\n======= writing test result file is done in {} ! ===========".format(test_result_file_path))
    print("\n\n=============== This is result with dict('{}')==================".format(dict_file_path))
    print("\n====== Test set('{}')'s Results ======".format(files_sorted))
    print("The total of sentences: {}, the 1/3: {}".format(len(raw_data), len(raw_data)/3))
    print("The total of words: {}".format(total_tok_num))
    print("The total of matched toks: {}".format(matched_tok_num))
    print("The total of not_matched_tok with dict('{}'): {}".format(dict_type, non_matched_tok_num_with_dict))
    print("The total of original word num matched without dict('{}'): {}".format(dict_type, original_word_num_matched_without_dict))
    print("The accuracy(the total of matched toks /the total of words) not considerting dictionary: {}".format(matched_tok_num/total_tok_num)) 
    print("The accuracy((the total of matched toks - the total of toks not in dict)/the total of words) considerting dictionary: {}".format((matched_tok_num - original_word_num_matched_without_dict)/total_tok_num))
    print("====== The aggregate is done ! ======")


 
if __name__ == "__main__":
    """

    ROOT_DIR = "corpus/"

    DICT_ROOT = "dict/"

    TEST_RESULT_ROOT = "test_result/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    # For Debugging option, increasing level enumerate log in detail
    DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

    ## if DEBUG_LEVEL is 0, No print for debugging
    DEBUG = DEBUG_LEVEL[0]

    ## train and test corpus 
    LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string
    TEST_LINE_OPT = "# output = "

    TRAIN_CORPUS = "train.txt"
    TEST_CORPUS = "test.txt"

    START_TOK = "<START_TOK>"
    END_TOK = "<END_TOK>"

    ## Dictionary type 
    ##             0   ,     1    ,      2    ,     3     ,      4     ,               5            ,                 6
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

    DICT_HEAD =  ["N-gram_word","Heat-tail_Token","Fre.","Prob."]

    """

    dict_types = DICT_TYPES[3] #

    dict_child_dic = CHILD_DIR[0] #

    test_child_dict = CHILD_DIR[0] #

    ## For Test
    test_dataset_loc = ROOT_DIR+TEST_CORPUS #test_child_dict+TEST_CORPUS

    dict_path = DICT_ROOT+dict_types #test_child_dict+dict_types

    test_result_file_loc = TEST_RESULT_ROOT+test_child_dict[:-1]+"_with_"+dict_types+"_Dict_result.txt" #child_dict+child_dict[:-1]+"_with_"+dict_types+"_Dict_result.txt"

    ## For Experiments
    ###test_dataset_loc = ROOT_DIR+test_child_dict+TEST_CORPUS

    ###dict_path = DICT_ROOT+dict_child_dic+dict_types
   
    ###test_result_file_loc = TEST_RESULT_ROOT+test_child_dict+test_child_dict[:-1]+"_with_"+dict_types+"_Dict_result.txt"

    test(test_dataset_loc, dict_path, test_result_file_loc, dict_type=dict_types)
