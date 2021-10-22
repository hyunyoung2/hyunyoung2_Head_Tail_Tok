"""
With head-tail corpus generated using https://github.com/hyunyoung2/Hyunyoung_HEAD_TAIL_CORPUS

with BUFS-JBNUCorpus2020 from https://github.com/bufsnlp2030/BUFS-JBNUCorpus2020

This code splits data into train and test. 

the structure of head-tail corpus is as follows:

# sent_id = 2002
# file = 00002
# text = 그 결과에 따라 대통령의 국정원 개혁 의지를 판단할 것
1	그	그	그	DET	MM	MM
2	결과에	결과+에	결과+에	NOUN	NNG+JKB	NNG+JKB
3	따라	따라	따르+아	VERB	VV_EC	VV+EC
4	대통령의	대통령+의	대통령+의	NOUN	NNG+JKG	NNG+JKG
5	국정원	국정원	국정원	NOUN	NNG	NNG
6	개혁	개혁	개혁	NOUN	NNG	NNG
7	의지를	의지+를	의지+를	NOUN	NNG+JKO	NNG+JKO
8	판단할	판단할	판단하+ㄹ	VERB	VV_ETM	VV+ETM
9	것	것	것	NOUN	NNB	NNB

# sent_id = 2003
# file = 00002
# text = 금리가 연 30% 이상이라는 말에 놀라면서도 상당수 고객이 대출 신청을 한다 .
1	금리가	금리+가	금리+가	NOUN	NNG+JKS	NNG+JKS
2	연	연	연	NOUN	NNG	NNG
3	30%	30+%	30+%	NUM	SN+SW	SN+SW
4	이상이라는	이상+이라는	이상+이+라는	VERB	NNG+VCP_ETM	NNG+VCP+ETM
5	말에	말+에	말+에	NOUN	NNG+JKB	NNG+JKB
6	놀라면서도	놀라+면서도	놀라+면서+도	VERB	VV+EC_JX	VV+EC+JX
7	상당수	상당수	상당수	NOUN	NNG	NNG
8	고객이	고객+이	고객+이	NOUN	NNG+JKS	NNG+JKS
9	대출	대출	대출	NOUN	NNG	NNG
10	신청을	신청+을	신청+을	NOUN	NNG+JKO	NNG+JKO
11	한다	한+다	하+ㄴ다	VERB	VV+EF	VV+EF
12	.	.	.	PUNC	SF	SF


For the colums of head-tail corpu 

NEW_HEAD = ["ID", "FROM", "HEAD_TAIL", "LEMMA", "UPOS", "HEAD_TAIL_XPOS", "XPOS"]

For train and test, 

There are three pairs of train and test, which are kcc_q28_only, kcc_150_only, kcc_150_n_q28
"""

#-*- coding: utf-8 -*-

from glob import glob
import os
from collections import OrderedDict
from random import shuffle

ROOT_DIR = "dataset/"

## 00_total_data directory has 00_kcc_q28_only and 01_kcc_150_only

PARENT_DIR = ["00_total_data/", "01_train/", "02_test/"]

CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

# For Debugging option, increasing level enumerate log in detail
DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

## if DEBUG_LEVEL is 0, No print for debugging
DEBUG = DEBUG_LEVEL[0]

# For lines,
LINE_OPTS = ["# sent_id", "# file", "# text", ''] # "" means empty string

## train and test corpus 
NEW_LINE_FORMAT = ["# text =", "# head_tail_text ="]

##                    0  ,    1  ,      2     ,    3   ,   4   ,        5        ,    6 
##
HEAD_TAIL_COLUMNS = ["ID", "FROM", "HEAD_TAIL", "LEMMA", "UPOS", "HEAD_TAIL_XPOS", "XPOS"]

def check_columns(data):
    """Check if the lines in input corpus is columns or not 

    the input file's format is the following:
 
    # sent_id = 2003
    # file = 00002
    # text = 금리가 연 30% 이상이라는 말에 놀라면서도 상당수 고객이 대출 신청을 한다 .
    1	금리가	금리+가	금리+가	NOUN	NNG+JKS	NNG+JKS
    2	연	연	연	NOUN	NNG	NNG
    3	30%	30+%	30+%	NUM	SN+SW	SN+SW
    4	이상이라는	이상+이라는	이상+이+라는	VERB	NNG+VCP_ETM	NNG+VCP+ETM
    5	말에	말+에	말+에	NOUN	NNG+JKB	NNG+JKB
    6	놀라면서도	놀라+면서도	놀라+면서+도	VERB	VV+EC_JX	VV+EC+JX
    7	상당수	상당수	상당수	NOUN	NNG	NNG
    8	고객이	고객+이	고객+이	NOUN	NNG+JKS	NNG+JKS
    9	대출	대출	대출	NOUN	NNG	NNG
    10	신청을	신청+을	신청+을	NOUN	NNG+JKO	NNG+JKO
    11	한다	한+다	하+ㄴ다	VERB	VV+EF	VV+EF
    12	.	.	.	PUNC	SF	SF

    As you can see, to pick up columns starting from ID number

    The colums is  HEAD_TAIL_COLUMNS = ["ID", "FROM", "HEAD_TAIL", "LEMMA", "UPOS", "HEAD_TAIL_XPOS", "XPOS"]

    with LINE_OPTS = ["# sent_id", "# file", "# text", ''] # '' means empty string
     
    If input matches any of LINE_OPTS, this function returns True
   
    If not, return False

    Arg:
       data(str): Text lines read from corpus 

    Return:
       var(boolean): If input matches any of LINE_OPTS, this function returns True
                     If not, return False
    """
    assert isinstance(data, str), "The type of input is wrong in 'check_columns' function: the type of data is {}, the value is {}".format(type(data), data)

    if LINE_OPTS[0] in data:
        return True
    elif LINE_OPTS[1] in data:
        return True
    elif LINE_OPTS[2] in data:
        return True
    elif LINE_OPTS[3] == data:
        return True
    else:
        return False

def read_raw_corpus(path):
    """Reading data line by line from head-tail copurs generated from https://github.com/hyunyoung2/Hyunyoung_HEAD_TAIL_CORPUS

    with BUFS-JBNUCorpus2020 from https://github.com/bufsnlp2030/BUFS-JBNUCorpus2020

    Arg:
      path(str): The path to a raw corpus to be read 
                 Normally, files in the directory ori_data 
    return:
      data(list): A list of lines in raw corpus like 
                  ["line1_str", "line2_str", ..., "line_n_str"]
    """

    assert isinstance(path, str), "The type of input is wrong 'read_raw_corpus' function: the type of path is {}, the value is {}".format(type(path), path)


    with open(path, "r") as wr:
        data = [val.strip() for val in wr.readlines()] # to strip the front and back from the text

    # After stripping the element in KCC POS corpus, 
    # check the number of columns is 10
    for idx, val in enumerate(data):
        if check_columns(val): # check the column with LINE_OPTS = ["# sent_id", "# file", "# text", ''] # '' means empty string
            continue
        else:  
            temp = val.split("\t")
            ## checking if the number of LEMA and XPOS column matches 
            assert len(temp[1]) == len("".join(temp[2].split("+"))), "FROM != HEAD_TAIL in HEAD-TAIL Corpus, FROM('{}') and HEAD_TAIL('{}')".format(temp[1], temp[2])
            ## Checking if the number of columns from the KCC tagged corpus is 10
            assert len(temp) == 7, "KCC corpus have an error, check the number of columns\n, the error: {}".format(val)
      

    if DEBUG in DEBUG_LEVEL[1:]:
        print("\n===== Reading a file of {} =====".format(path))
        print("The number of lines: {}".format(len(data)))
        print("for top 5 of examples, \n{}".format(data[0:15]))

    return data

def count_sents_in_file(file_data):
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
    ###tokens_num = 0
 

    # For lines,
    ##LINE_OPTS = ["# sent_id", "# file", "# text", ''] # "" means empty string
    for idx, val in enumerate(file_data):
        if LINE_OPTS[2] in val: # original sentence
            sent = val.split()[3:]

            lines_num += 1

            #tokens_num += len(sent)
         
         
    if DEBUG in DEBUG_LEVEL[0:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check count_tok_and_line_num function =====")
        print("The total of sents: {}".format(lines_num))

    return lines_num 

 
def counting_train_and_test(train_or_test_path):
    """extract from train and test directory with the following format

    # text = 
    # Head_tail_text =

    with NEW_LINE_FORMAT = ["# text =", "# head_tail_text ="]

    From   

    ROOT_DIR = "dataset/"

    PARENT_DIR = ["00_total_data/", "01_train/", "02_test/"]

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    to 

    ../corpus/CHILD_DIR
    
    ../corpus/CHILD_DIR

    Args:
        total_data_dir(str): the position of total data 

    Return:
        None
    """
    list_of_files = glob(train_or_test_path+"*", recursive=True)
     
    files_sorted = sorted(list_of_files)

    text_dict = OrderedDict()

    num_sents = []

    total_sents_num = 0

    for idx, val in enumerate(files_sorted):
        print("\n===== Preprocessing the file {} under directory {} =====".format(val, train_or_test_path))

        raw_data = read_raw_corpus(val)

        raw_sent_num = count_sents_in_file(raw_data)

        total_sents_num += raw_sent_num

        num_sents.append(raw_sent_num)
        print("\n===== Sent num in file {} =====".format(val))
        print("The number of sents: {}".format(raw_sent_num))

           
    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== a list of files to be read =====")
        print("The type of 'list_of_files': {}".format(type(list_of_files)))
        print("The number of files: {}".format(len(list_of_files)))
        print("For top 5 of examples, \n{}".format(list_of_files[0:5]))
        print("\n===== a list of files is sorted =====")
        print("The type of 'files_sorted': {}".format(type(files_sorted)))
        print("The number of files sorted: {}".format(len(files_sorted)))
        print("For top 3 of examples sorted: {}".format(files_sorted[0:5]))
        print("\n===== counting sents =====")
        print("The average of sent: {}".format(total_sents_num/len(files_sorted)))
        a = sorted(num_sents)
        print("The total of sents num: {}".format(total_sents_num))
        print("The maximum of sents num: {}".format(a[-1]))
        print("The minimum of sents num: {}".format(a[0]))
        print("sents_num array: \n{}".format(a[:100]))
                                 
    print("\n===== counting is done for dicrectory {} ! =====".format(train_or_test_path))


if __name__ == "__main__":

    ## 00_total_data directory has 00_kcc_q28_only and 01_kcc_150_only

    ##PARENT_DIR = ["00_total_data/", "01_train/", "02_test/"]

    ##CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]


    ## KCC Q28
    ##counting_train_and_test(ROOT_DIR+PARENT_DIR[0]+CHILD_DIR[0])
    ##input()
    ## KCC 150 
    counting_train_and_test(ROOT_DIR+PARENT_DIR[0]+CHILD_DIR[1])
