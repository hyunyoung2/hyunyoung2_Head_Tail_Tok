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

def write_file(path, data):
    """Write head-tail corpus splitting into train and test


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

    Args:
        path(str) : the destination to write with data
        data(list) : data to construct Head-Tail corpus 
    Return:
        None  
    """

    with open(path, "w") as fw:
        for idx, val in enumerate(data):
            if LINE_OPTS[3] == val:
                fw.write("\n")
            else:
                fw.write(val+"\n")

def split_data(total_data_dir, ratio=0.8):
    """split total data into train and test
   
    When splitting data, the standard is file

    the total of filse in a directory are split into a ration, which is training 8 and 2 by default

    ROOT_DIR = "dataset/"

    PARENT_DIR = ["00_total_data/", "01_train/", "02_test/"]

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    Args:
        total_data_dir(str): the position of total data 

    Return:
        None
    """
   
    list_of_files = glob(total_data_dir+"*", recursive=True)
     
    files_sorted = sorted(list_of_files)

    ## total of files
    files_num = len(files_sorted)

    ## The number of training data, total of data - the number of trainin = the number of test
    train_num = round(files_num * ratio)

    files_idx = list(range(len(files_sorted)))

    ## To radomly extract files
    shuffle(files_idx)


    ## Training data
    train_idx = []

    ## Test data
    test_idx = []

    ## split indices into train and test 
    for idx in files_idx:
          
        if len(train_idx) <= train_num:
            train_idx.append(idx)
        else:
            test_idx.append(idx)
    
    for idx, val in enumerate(files_sorted):
        print("\n===== Preprocessing the file {} under directory {} =====".format(val, total_data_dir))

        file_loc = val.split("/")

        raw_data = read_raw_corpus(val)
       
        if idx in train_idx:
            write_file(file_loc[0]+"/"+PARENT_DIR[1]+"/"+file_loc[2]+"/"+file_loc[3], raw_data)
        elif idx in test_idx:
            write_file(file_loc[0]+"/"+PARENT_DIR[2]+"/"+file_loc[2]+"/"+file_loc[3], raw_data)
           
    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== a list of files to be read =====")
        print("The type of 'list_of_files': {}".format(type(list_of_files)))
        print("The number of files: {}".format(len(list_of_files)))
        print("For top 5 of examples, \n{}".format(list_of_files[0:5]))
        print("\n===== a list of files is sorted =====")
        print("The type of 'files_sorted': {}".format(type(files_sorted)))
        print("The number of files sorted: {}".format(len(files_sorted)))
        print("For top 3 of examples sorted: {}".format(files_sorted[0:5]))
        print("\n===== train and test indices =====")
        print("The suffled top 5 indices: {}, {}".format(len(files_idx), files_idx[0:5]))
        print("The top 5 train indices: {}, {}".format(len(train_idx), train_idx[0:5]))
        print("The top 5 test indices: {}, {}".format(len(test_idx), test_idx[0:5]))

    print("\n===== Splitting into train and test is done ! =====")


def split_function_call():
    """intermediate function to call function named 'split_data'.

    with the structure of directory as follows:

    ROOT_DIR = "dataset/"

    PARENT_DIR = ["00_total_data/", "01_train/", "02_test/"]

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]
    """

    ## KCC Q28
    split_data(ROOT_DIR+PARENT_DIR[0]+CHILD_DIR[0], ratio=0.9) #0.8)

    ##input()

    ## KCC 150
    split_data(ROOT_DIR+PARENT_DIR[0]+CHILD_DIR[1], ratio=0.9) #0.8)


def extract_text(file_data, pos_option=True):
    """This function extract sentences, words, head-tail tokens in a file 

    with the following format

    # text = 
    # Head_tail_text =

    ## train and test corpus 
    with NEW_LINE_FORMAT = ["# text =", "# head_tail_text ="]
    
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

    This function extract sentences, words, head-tail tokens in a file 

    Args:
        file_data(list): this is data read from head-tail corpus line by line
        pos_option(bool): If you want to make head-tail corpus wiht pos, pos_option is True
                          If not, pos_option is False

    Rerturns:
        sents(list): the list of sentences in a file
        head_tail_texts(list): the list of text from  head_tail_tokens with '# head_tail_text =' in a file 
                               with pos tagging or without pos tagging on head-tail token
 
    """

    # For lines,
    # with LINE_OPTS = ["# sent_id", "# file", "# text", ''] # "" means empty string

    # train and test corpus 
    #with NEW_LINE_FORMAT = ["# text =", "# head_tail_text ="]
 
    sents = []
    head_tail_texts = []
 
    sent_idx = 0 
    for idx, val in enumerate(file_data):
        if LINE_OPTS[0] in val:
            continue
        elif LINE_OPTS[1] in val:
            continue
        elif LINE_OPTS[2] in val:
            sents.append(val)
            temp_head_tail_toks = [] 

        elif LINE_OPTS[3] == val:
            ## checking 
            ## the number of words in sentences is the same
            ## from both word and head_tail colums
            new_line_fmt = NEW_LINE_FORMAT[1].split()
            
            temp_head_tail_txt =  new_line_fmt + temp_head_tail_toks

            temp_sent = sents[sent_idx].split()
            assert len(temp_sent) == len(temp_head_tail_txt), "In the 'extract_text' function, len(temp_sent) != len(temp_head_tail_txt), temp_sent('{}', '{}') and temp_head_tail_txt('{}', '{}')".format(len(temp_sent), temp_sent, len(temp_head_tail_txt), temp_head_tail_txt)

            ## check words in a sentence is the same from 
            for _idx, _val in enumerate(temp_sent):
                if _idx < 3:
                    continue
                else:
                    if pos_option == False:
                        assert _val == "".join(temp_head_tail_txt[_idx].split('+')), "In the e' function, _val != temp_head_tail_txt[_idx], val('{}') and temp_head_tail_txt['{}']('{}')".format(val, _idx, temp_head_tail_txt[_idx])
                    ### If you want to check the number of character in a word with pos tagging
                    ### write down here assert statment for debugging

            head_tail_texts.append(" ".join(temp_head_tail_txt))

            ##print("Temp_head_tail_tokens: {}, {}".format(len(temp_head_tail_toks), temp_head_tail_toks))
            ##print("Temp_head_tail_txt: {}, {}".format(len(temp_head_tail_txt), temp_head_tail_txt))
            ##print("Sents: {}, {}".format(len(sents), sents))
            ##print("temp_sent: {}, {}".format(len(temp_sent), temp_sent))
            ##input()

            temp_head_tail_toks =[]
            sent_idx +=1
        else: ## columns 2	결과에	결과+에	결과+에	NOUN	NNG+JKB	NNG+JKB
            ## HEAD_TAIL_COLUMNS = ["ID", "FROM", "HEAD_TAIL", "LEMMA", "UPOS", "HEAD_TAIL_XPOS", "XPOS"]
            col = val.split()

            if pos_option == True:
                ## HEAD_TAIL_COLUMNS = ["ID", "FROM", "HEAD_TAIL", "LEMMA", "UPOS", "HEAD_TAIL_XPOS", "XPOS"]
                ## 결과/NNG+에/JKB
                temp_hd = col[2].split("+")
                temp_hd_pos = col[-2].split("+")
                assert len(temp_hd) == len(temp_hd_pos), "len(tmep_hd) != len(temp_hd_pos), len(temp_hd): {} and len(temp_hd_pos): {}".format(len(temp_hd), len(temp_hd_pos))

                temp_pos_tok = []
                for i in range(len(temp_hd)):
                    temp_pos_tok.append(temp_hd[i]+"/"+temp_hd_pos[i])

                temp_head_tail_toks.append("+".join(temp_pos_tok))

            else:
                ## 결과+에
                temp_head_tail_toks.append(col[2])

    if DEBUG in DEBUG_LEVEL[0:]: # ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]
        print("\n===== Check handle_a_file function =====")
        print("The number of sents: {}".format(len(sents)))
        print("For top 5 examples, {}".format(sents[:5]))
        print("The number of head-tail texts: {}".format(len(head_tail_texts)))
        print("For top 5 examples, {}".format(head_tail_texts[:5])) 
        print("The fianl sent idx: {}".format(sent_idx))

    return sents, head_tail_texts 

def write_train_and_test_file(path, dict_data):
    """Write train and test datat for head-tail corpus

    with new format

    # train and test corpus 
    # with NEW_LINE_FORMAT = ["# text =", "# head_tail_text ="]
 
    # text = 
    # Head_tail_text =

    Args:
        path(str) : the destination to write with data
        data(dict) : data to construct Head-Tail corpus 
    Return:
        None  
    """

    with open(path, "w") as wf:
        for idx, val in enumerate(dict_data.items()):
            wf.write(val[0][0]+"\n")
            wf.write(val[0][1]+"\n")
            wf.write("\n") # Empty string
 
def extract_train_and_test(train_or_test_path, pos_opt=False):
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
        pos_opt(bool); if True, make head-tail corpus with pos
                       if Fals, make head-tail corpus without pos

    Return:
        None
    """
    list_of_files = glob(train_or_test_path+"*", recursive=True)
     
    files_sorted = sorted(list_of_files)

    text_dict = OrderedDict()

    for idx, val in enumerate(files_sorted):
        print("\n===== Preprocessing the file {} under directory {} =====".format(val, train_or_test_path))

        file_loc = val.split("/")

        raw_data = read_raw_corpus(val)

        raw_sents, raw_head_tail_texts = extract_text(raw_data, pos_opt)

        assert len(raw_sents) == len(raw_head_tail_texts), "len(raw_sents) != len(raw_head_tail_texts), raw_sents('{}'):{} and raw_head_tail_texts('{}'):{}".format(len(raw_sents), raw_sents, len(raw_head_tail_texts), raw_head_tail_texts)
        for sent_idx, sent_val in enumerate(raw_sents):
            _key = (sent_val, raw_head_tail_texts[sent_idx])
            if text_dict.get(_key):
                text_dict[_key] += 1
            else:
                text_dict[_key] = 1

    des_dir = "../corpus/"+file_loc[2]+"/"

    if file_loc[1] == PARENT_DIR[1][:-1]: # "01_train/"
        final_file = des_dir+"train.txt"
    elif file_loc[1] == PARENT_DIR[2][:-1]: # "02_test/"
        final_file = des_dir+"test.txt"
    else:
        assert False,  "your des_dir and final_file is wrong"

         
    write_train_and_test_file(final_file, text_dict)

           
    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== a list of files to be read =====")
        print("The destination of dic: {}".format(des_dir))
        print("The file name: {}".format(final_file))
        print("The type of 'list_of_files': {}".format(type(list_of_files)))
        print("The number of files: {}".format(len(list_of_files)))
        print("For top 5 of examples, \n{}".format(list_of_files[0:5]))
        print("\n===== a list of files is sorted =====")
        print("The type of 'files_sorted': {}".format(type(files_sorted)))
        print("The number of files sorted: {}".format(len(files_sorted)))
        print("For top 3 of examples sorted: {}".format(files_sorted[0:5]))
        print("\n===== text dict =====")
        print("The samples of text_dict: {}\n{}".format(len(list(text_dict.items())), list(text_dict.items())[0:5]))


    print("\n===== extract text and head_tail_text into train and test is done ! =====")


def extract_function_call(pos_tag=False):
    """intermediate function to call function named 'extract_train_and_test'.

    with the structure of directory as follows:

    ROOT_DIR = "dataset/"

    PARENT_DIR = ["00_total_data/", "01_train/", "02_test/"]

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]
    """

    ## KCC Q28 Train
    extract_train_and_test(ROOT_DIR+PARENT_DIR[1]+CHILD_DIR[0], pos_tag)
    ##input()
    ## KCC Q28 Test
    extract_train_and_test(ROOT_DIR+PARENT_DIR[2]+CHILD_DIR[0], pos_tag)

    ##input()

    ## KCC 150 Train
    extract_train_and_test(ROOT_DIR+PARENT_DIR[1]+CHILD_DIR[1], pos_tag)
    ##input()
    ## KCC 150 Test
    extract_train_and_test(ROOT_DIR+PARENT_DIR[2]+CHILD_DIR[1], pos_tag)

    ##input()

    ## KCC 150_N_Q28 Train
    extract_train_and_test(ROOT_DIR+PARENT_DIR[1]+CHILD_DIR[2], pos_tag)
    ##input()
    ## KCC 150_N_Q28 Test
    extract_train_and_test(ROOT_DIR+PARENT_DIR[2]+CHILD_DIR[2], pos_tag)


if __name__ == "__main__":

    ##print("\n===== Splitting into train and test is on =====")
    ##if True: # split into train and test
    ##    split_function_call()

    ##print("\n===== Extracting train and test is on =====")
    if True: # extract train and test
        ## With pos tagging
        ##extract_function_call(pos_tag=True)
        ## without pos tagging
        extract_function_call(pos_tag=False)
