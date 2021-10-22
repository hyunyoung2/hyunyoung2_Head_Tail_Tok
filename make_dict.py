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


def extract_ngram(file_data, dictionary_type="None"):
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


    ## dictionary type 
    ##                 0  ,     1    ,      2    ,      3    ,      4     ,              5             ,                 6        
    ## DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]
 
    Args:
       file_data(list): sentences read corpus line by line
       dictionary_type(str): The type of ngram dictionary  

    Returns:
       uni_word(list): pairs of (unigram FROM, HEAD-TAIL-TOKEN)

       left_bi_word(list): pairs of (left bigram FROM, HEAD-TAIL-TOKEN)
       left_tri_word(list): pairs of (left trigram FROM, HEAd-TAIL-TOKEN)
    
       right_bi_word: pairs of (right bigram FROM, HEAD-TAIL-TOKEN) 
       right_tri_word: pairs of (right trigram FROM, HEAD-TAIL-TOKEN)

       bidir_word_win_1: pairs of (left and right bigram FROM, HEAD-TAIL-TOKEN) # left 1 and right 1
       bidir_word_win_2: pairs of (left and right trigram FROM, HEAD-TAIL-TOKEN) # left 2 and right 2
    """

    assert dictionary_type in DICT_TYPES, "the type of dictionary in 'extract_ngram' function is wrong. dictionary_type: {}".format(dictionary_type)

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
    
    ## LINE_OPTS = ["# text =", "# head_tail_text =", ""] # "" means empty string
    sent = []
    hetail = []
    for idx, val in enumerate(file_data):
        if LINE_OPTS[0] in val: # original sentence
            sent = val.split()[3:]

            # double checking empty list 
            assert sent != [], "you sent is empty list: {}".format(sent)
        elif LINE_OPTS[1] in val: # head-tail sentence
            hetail = val.split()[3:]

            # double checking empty list 
            assert hetail != [], "you heatil is empty list: {}".format(hetail)

        elif LINE_OPTS[2] == val:

            len_sent = len(sent)
            
            for temp_sent_idx, temp_sent_val in enumerate(sent):
                
                # uni_word
                uni_hetail = hetail[temp_sent_idx]


                ## dictionary type 
                ##                 0  ,     1    ,      2    ,      3    ,      4     ,              5             ,                 6        
                ## DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]
                ## Later on, If you want to change the generation of dictionary into two type or more, 
                ## Change dictionary == DICT_TYPES[#] 
 
                if dictionary_type == DICT_TYPES[0]:
                    uni_pair = (temp_sent_val, uni_hetail)
                
                    before_len_uni_word = len(uni_word)
 
                    uni_word.append(uni_pair)

                    after_len_uni_word = len(uni_word)

                    assert before_len_uni_word < after_len_uni_word, "You uni_word is wrong, before('{}') and after('{}')".format(before_len_uni_word, after_len_uni_word)

                elif dictionary_type == DICT_TYPES[1]:
             
                    before_len_left_bi = len(left_bi_word)
                    ## left section
                    # left bi 
                    if temp_sent_idx == 0:
                        left_bi = ((START_TOK, temp_sent_val), uni_hetail)
                        left_bi_word.append(left_bi)
                    else: ## temp_sent_idx != 0:
                        left_bi = ((sent[temp_sent_idx-1], temp_sent_val), uni_hetail)
                        left_bi_word.append(left_bi)

                    after_len_left_bi = len(left_bi_word)

                    assert before_len_left_bi < after_len_left_bi, "You left_bi_word is wrong, before('{}') and after('{}')".format(before_len_left_bi, after_len_left_bi)
                
                elif dictionary_type == DICT_TYPES[2]:
                    before_len_left_tri = len(left_tri_word)
                    # left tri 
                    if temp_sent_idx == 0:
                        left_tri = ((START_TOK, START_TOK, temp_sent_val), uni_hetail)
                        left_tri_word.append(left_tri)
                    elif temp_sent_idx == 1:
                        left_tri = ((START_TOK, sent[temp_sent_idx-1], temp_sent_val), uni_hetail)
                        left_tri_word.append(left_tri)
                    else: # 1 < temp_sent_idx and temp_sent_idx < len(sent):
                        left_tri = ((sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val), uni_hetail)
                        left_tri_word.append(left_tri)
                 
                    after_len_left_tri = len(left_tri_word)

                    assert before_len_left_tri < after_len_left_tri, "You left_tri_word is wrong, before('{}') and after('{}')".format(before_len_left_tri, after_len_left_tri)

                elif dictionary_type == DICT_TYPES[3]:        
                    # right section
                    reverse_sent_idx = (temp_sent_idx+1) * -1

                    reverse_uni_hetail = hetail[reverse_sent_idx]
       
                    before_len_right_bi = len(right_bi_word)
                    # right bi 
                    if reverse_sent_idx == -1:
                        right_bi = ((sent[reverse_sent_idx], END_TOK), reverse_uni_hetail)
                        right_bi_word.append(right_bi)
                    elif len_sent * -1 <= reverse_sent_idx and reverse_sent_idx < -1:
                        right_bi = ((sent[reverse_sent_idx], sent[reverse_sent_idx+1]), reverse_uni_hetail)
                        right_bi_word.append(right_bi)
                    else:
                        assert False, "Reveres_sent_idx < len(sent) * -1 or Reverse_sent_idx > -1 for right bi"
 
                    after_len_right_bi = len(right_bi_word)
    
                    assert before_len_right_bi < after_len_right_bi, "You right_bi_word is wrong, before('{}') and after('{}')".format(before_len_right_bi, after_len_right_bi)

                elif dictionary_type == DICT_TYPES[4]:                
                    # right section
                    reverse_sent_idx = (temp_sent_idx+1) * -1

                    reverse_uni_hetail = hetail[reverse_sent_idx]
       
                    before_len_right_tri = len(right_tri_word)
                    # right tri
                    if reverse_sent_idx == -1:
                       right_tri = ((sent[reverse_sent_idx], END_TOK, END_TOK), reverse_uni_hetail)
                       right_tri_word.append(right_tri)
                    elif reverse_sent_idx == -2:
                       right_tri = ((sent[reverse_sent_idx], sent[reverse_sent_idx+1], END_TOK), reverse_uni_hetail)
                       right_tri_word.append(right_tri)
                    elif len_sent * -1 <= reverse_sent_idx and reverse_sent_idx < -2:
                       right_tri = ((sent[reverse_sent_idx], sent[reverse_sent_idx+1], sent[reverse_sent_idx+2]), reverse_uni_hetail)
                       right_tri_word.append(right_tri)
                    else:
                       assert False, "Reverse_sent_idx < len(sent) * -1 or Reverse_sent_idx > -1 for right tri"

                    after_len_right_tri = len(right_tri_word)
               
                    assert before_len_right_tri < after_len_right_tri, "You right_tri_word is wrong, before('{}') and after('{}')".format(before_len_right_tri, after_len_right_tri)

                elif dictionary_type == DICT_TYPES[5]: 
                    # bidirectional section
                    # bidirectional word with window 1 
                    before_len_bi_win_1 = len(bidir_word_win_1)
                    if len_sent == 1:
                        bi_word_window_1 = ((START_TOK, temp_sent_val, END_TOK), uni_hetail)
                        bidir_word_win_1.append(bi_word_window_1)
                    else:
                        if temp_sent_idx == 0:
                            bi_word_window_1 = ((START_TOK, temp_sent_val, sent[temp_sent_idx+1]), uni_hetail)
                            bidir_word_win_1.append(bi_word_window_1)
                        elif 0 < temp_sent_idx and temp_sent_idx < len_sent - 1:
                            bi_word_window_1 = ((sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1]), uni_hetail)
                            bidir_word_win_1.append(bi_word_window_1)
                        elif temp_sent_idx == len_sent -1: 
                            bi_word_window_1 = ((sent[temp_sent_idx-1], temp_sent_val, END_TOK), uni_hetail)
                            bidir_word_win_1.append(bi_word_window_1)
                        else:
                            assert False, "temp_sent_idx < 0, for bidirectional word with window 1"
 
                    after_len_bi_win_1 = len(bidir_word_win_1)

                    assert before_len_bi_win_1 < after_len_bi_win_1, "You bidir_word_win_1 is wrong, before('{}') and after('{}')".format(before_len_bi_win_1, after_len_bi_win_1)

                elif dictionary_type == DICT_TYPES[6]:
                    # bidirectional word with window 2 

                    before_len_bi_win_2 = len(bidir_word_win_2)

                    if len_sent == 1:
                        bi_word_window_2 = ((START_TOK, START_TOK, temp_sent_val, END_TOK, END_TOK), uni_hetail)
                        bidir_word_win_2.append(bi_word_window_2) 
                    elif len_sent == 2:
                        if temp_sent_idx == 0:
                            bi_word_window_2 = ((START_TOK, START_TOK, temp_sent_val, sent[temp_sent_idx+1], END_TOK), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        elif temp_sent_idx == 1:
                            bi_word_window_2 = ((START_TOK, sent[temp_sent_idx-1], temp_sent_val, END_TOK, END_TOK), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                    elif len_sent == 3:
                        if temp_sent_idx == 0:
                            bi_word_window_2 = ((START_TOK, START_TOK, temp_sent_val, sent[temp_sent_idx+1], sent[temp_sent_idx+2]), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        elif temp_sent_idx == 1:
                            bi_word_window_2 = ((START_TOK, sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1], END_TOK), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        elif temp_sent_idx == 2:
                            bi_word_window_2 = ((sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val, END_TOK, END_TOK), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)    
                    else: # len_sent > 3
                        if temp_sent_idx == 0:
                            bi_word_window_2 = ((START_TOK, START_TOK, temp_sent_val, sent[temp_sent_idx+1], sent[temp_sent_idx+2]), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        elif temp_sent_idx == 1:
                            bi_word_window_2 = ((START_TOK, sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1], sent[temp_sent_idx+2]), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        elif 1 < temp_sent_idx and temp_sent_idx < len_sent - 2:
                            bi_word_window_2 = ((sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val , sent[temp_sent_idx+1], sent[temp_sent_idx+2]), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        elif temp_sent_idx == len_sent -2: 
                            bi_word_window_2 = ((sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val, sent[temp_sent_idx+1], END_TOK), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        elif temp_sent_idx == len_sent -1:
                            bi_word_window_2 = ((sent[temp_sent_idx-2], sent[temp_sent_idx-1], temp_sent_val, END_TOK, END_TOK), uni_hetail)
                            bidir_word_win_2.append(bi_word_window_2)
                        else:
                            assert False, "temp_sent_idx < 0, for bidirectional word with window 2"

                    after_len_bi_win_2 = len(bidir_word_win_2)

                    assert before_len_bi_win_2 < after_len_bi_win_2, "You bidir_word_win_2 is wrong, before('{}') and after('{}')".format(before_len_bi_win_2, after_len_bi_win_2)
                else:
                    assert False, "When you use 'extract ngram' function, dictionary type is {}".format(dictionary_type) 
                                
                         
            ## DEBUGGING 
            ##print("\nSent (len, elements): {}\n{}".format(len(sent), sent))
            ##print("hetail (len, elements): {}\n{}".format(len(hetail), hetail))
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
            
            sent = []
            hetail = []

        else:
            assert False, "Extract ngram function is wrong, you need to check it"
        
        
    if DEBUG in DEBUG_LEVEL[0:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check extract ngram function with dictionary type {} =====".format(dictionary_type))
        print("The total sents: {}, 1/3 ration: {}".format(len(file_data), len(file_data)/3))
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

def count_pair_dict(data):
    """counting pairs of n-gram and head_tail

    Arg:
      data(list): this list consits of pairs of (n-gram, head_tail token)

    Return:
      pair_dict(dict): pair dictionary
                       {n-gram_0: {head_tail_token_1:cnt, head_tail_token_2:cnt}, ..., 
                        n_gram_1: {head_tail_token_1:cnt, head_tail_token_2:cnt}}
    """

    pair_dict = OrderedDict()

    for idx, val in enumerate(data):

        _key = val[0]
        _value = val[1]

        if pair_dict.get(_key):
            if pair_dict[_key].get(_value):
                pair_dict[_key][_value] += 1
            else:
                pair_dict[_key][_value] = 1
        else:
            pair_dict[_key] = OrderedDict()
            pair_dict[_key][_value] = 1

    if DEBUG in DEBUG_LEVEL[0:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check pair dict =====")
        print("The total pair_dict: {}".format(len(pair_dict)))
        print("The top 2 examples: {}".format(list(pair_dict.items())[0:5]))
       
    return pair_dict 

def count_ngram_word_dict(data):
    """counting pairs of n-gram and head_tail

    Arg:
      data(list): consits of pairs of (n-gram, head_tail token)

    Return:
      n_gram_word_dict(dict): double dictionary
                         {n-gram_0: cnt, ..., 
                          n_gram_1: cnt}
    """

    ngram_word_dict = OrderedDict()

    for idx, val in enumerate(data):

        _key = val[0]
        _ = val[1] # don't use
    
        if ngram_word_dict.get(_key):
            ngram_word_dict[_key] += 1
        else:
            ngram_word_dict[_key] = 1 

    if DEBUG in DEBUG_LEVEL[0:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check ngram word dict =====")
        print("The total ngram_word_dict: {}".format(len(ngram_word_dict)))
        print("The top 5 examples: {}".format(list(ngram_word_dict.items())[:5]))
 
    return ngram_word_dict

## this fuction will be removed
## prior knowledge is normally set to 1
def count_hetail_dict(data):
    """counting pairs of n-gram and head_tail


    prior probability

    So I have to set thie value to 1 since I don't know real value

    Arg:
      data(list): consits of pairs of (n-gram, head_tail token)

    Return:
      hetail_dict(dict): double dictionary
                         {head_tail_tok_0: cnt, ..., 
                          head_tail_tok_1: cnt}
    """

    hetail_dict = OrderedDict()

    for idx, val in enumerate(data):
        _ = val[0] # don't use
        _value = val[1] 
    
        if hetail_dict.get(_value):
            hetail_dict[_value] += 1
        else:
            hetail_dict[_value] = 1 

    if DEBUG in DEBUG_LEVEL[0:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check hetail dict =====")
        print("The total hetail_dict: {}".format(len(hetail_dict)))
        print("The top 5 examples: {}".format(list(hetail_dict.items())[:5]))
 
    return hetail_dict

def write_n_gram_dict_file(path, _word_dict, _pair_dict, dict_type="None"):
    """write n-gram dict into DICT_ROOT/CHILD_DIR[#]/uni_word_dict.txt

    DICT_ROOT = "dict/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    ## Dictionary type 
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

    DICT_HEAD =  ["N-gram_word","Heat-tail_Token","Fre.","Prob."]
    """

    # P(h|w) = c(w,h)/c(w) # uni_word

    # p(h|c) = c(c,h)/c(c) # bi, tri, and bidir_bi, bidir_tri

    assert dict_type in DICT_TYPES, "the type of dictionary in write_n_gram_dict_file is wrong. dict_type: {}".format(dict_type)

    with open(path, "w") as wf:

        wf.write("\t".join(DICT_HEAD)+"\n")
        for n_gram_word, hetail_pair in _pair_dict.items():

            if dict_type == DICT_TYPES[0]: ## "UNI"
                _ngram_key = n_gram_word
            else: 
                _ngram_key = " ".join(n_gram_word)

            ### Debugging
            ##print("\n===== write_n_gram_dict_file =====") 
            ##print("n_gram_word: {}".format(n_gram_word))
            ##print("_ngram_key: {}".format(_ngram_key))
            ##print("_word_dict[n_gram_word]: {}".format(_word_dict[n_gram_word]))
            ##print()
            ##print("hetail_pair: {}".format(list(hetail_pair.items())))
            hetail_pair_sorted = sorted(hetail_pair.items(), key=lambda x: x[1], reverse=True)
            ##print("hetail_pair_sorted('{}'): {}".format(type(hetail_pair_sorted), hetail_pair_sorted))
            ##input()
            ##print()
            for hetail_key, hetail_cnt in hetail_pair_sorted:
                ##print(hetail_key, hetail_cnt)
                ##input()
                wf.write(_ngram_key+"\t"+hetail_key+"\t"+str(hetail_cnt)+"\t"+str(hetail_cnt/_word_dict[n_gram_word]*100)+"\n")

    
    print("\n===== Writing n-gram dictionary is done in {} with type {} ======".format(path, dict_type))  
        
def make_dictionary(train_path, dict_file_path, dict_idx="None"):
    """Make dictrionary for head_tail tokens with train dataset

    using the structure of directory 
    
    ROOT_DIR = "corpus/"

    DICT_ROOT = "dict/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    TRAIN_CORPUS = "train.txt"
    TEST_CORPUS = "test.txt"

    ## Dictionary type 
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

   
    After counting n-gram From and head-tail token, write dictionary into DICT_ROOT+CHILD_DIR

    Args: 
       train_path(str): the location of train dataset
       dict_type(str): the type of dictionary of head-tail token
                       one of the followings

                       DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

    Return: 
       None 
    """

    assert dict_idx in DICT_TYPES, "making dictionary is wrong, chek dict_idx: {}".format(dict_idx)

    files_sorted = [train_path] ## If handling a variety of files

    assert len(files_sorted) == 1, "making dictionary is wrong!!, files_sorted: {}".format(files_sorted)

    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== a list of files to be read =====")
        print("The destination of dic: {}".format(train_path))
        print("The type of 'files_sorted': {}".format(type(files_sorted)))
        print("The number of files sorted: {}".format(len(files_sorted)))
        print("For examples sorted: {}".format(files_sorted))
 

    for idx, val in enumerate(files_sorted):
        print("\n===== Preprocessing the file {} =====".format(val))

        raw_data = read_raw_corpus(val)

        print("\n===== Reading the file {} is done ! =====".format(val))

        print("\n===== Counting n-gram('{}') starts in 'make_dictionary' function with 'extract_ngram' function =====".format(dict_idx)) 

        uni, left_bi, left_tri, right_bi, right_tri, bidir_bi_win_1, bidir_tri_win_2 =  extract_ngram(raw_data, dict_idx)
    
        print("\n===== Counting n-gram('{}') is done in 'make_dictionary' function with 'extract_ngram' function ! =====".format(dict_idx)) 

        ## dictionary type 
        ##                 0  ,     1    ,      2    ,      3    ,      4     ,              5             ,                 6        
        ## DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]
    
        ## this is not used, maybe I will remove this function   
        hetail_tok_dic = count_hetail_dict(uni) # this prior probability 
    
        if dict_idx == DICT_TYPES[0]: # uni dict
            ngram_and_hetail_pair_dict = count_pair_dict(uni)    
            ngram_dict = count_ngram_word_dict(uni)

        elif dict_idx == DICT_TYPES[1]: # left bi dict
            ngram_and_hetail_pair_dict = count_pair_dict(left_bi)    
            ngram_dict = count_ngram_word_dict(left_bi)

        elif dict_idx == DICT_TYPES[2]: # left tri dict
            ngram_and_hetail_pair_dict = count_pair_dict(left_tri)    
            ngram_dict = count_ngram_word_dict(left_tri)

        elif dict_idx == DICT_TYPES[3]: # right bi dict
            ngram_and_hetail_pair_dict = count_pair_dict(right_bi)    
            ngram_dict = count_ngram_word_dict(right_bi)


        elif dict_idx == DICT_TYPES[4]: # right tri dict
            ngram_and_hetail_pair_dict = count_pair_dict(right_tri)    
            ngram_dict = count_ngram_word_dict(right_tri)


        elif dict_idx == DICT_TYPES[5]: # bidirectional bi window 1 dict
            ngram_and_hetail_pair_dict = count_pair_dict(bidir_bi_win_1)    
            ngram_dict = count_ngram_word_dict(bidir_bi_win_1)


        elif dict_idx == DICT_TYPES[6]: # bidirectional tri window 2 dict
            ngram_and_hetail_pair_dict = count_pair_dict(bidir_tri_win_2)    
            ngram_dict = count_ngram_word_dict(bidir_tri_win_2)


        else:
            assert False, "Your dict type is wrong, the type of dict is {}".format(dict_type)

        ### I have to change this line if I want to experiment         
        write_n_gram_dict_file(path = dict_file_path, _word_dict = ngram_dict, _pair_dict = ngram_and_hetail_pair_dict, dict_type=dict_idx)

    
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

    dict_types = DICT_TYPES[3] 
   
    child_dict_type = CHILD_DIR[0]

    ## For Test
    train_dataset_loc = ROOT_DIR+TRAIN_CORPUS #child_dict_type+TRAIN_CORPUS

    dict_file_loc = DICT_ROOT+dict_types #child_dict_type+dict_types

    ## For Experiments
    ###train_dataset_loc = ROOT_DIR+child_dict_type+TRAIN_CORPUS

    ###dict_file_loc = DICT_ROOT+child_dict_type+dict_types

    make_dictionary(train_dataset_loc, dict_file_loc, dict_idx=dict_types)

