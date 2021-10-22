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

After head-tail tokenization, Check precision or recal and F1 score

"""
#-*- coding: utf-8 -*-

from glob import glob
import os
from collections import OrderedDict

TEST_RESULT_ROOT = "test_result/"

CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

# For Debugging option, increasing level enumerate log in detail
DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

## if DEBUG_LEVEL is 0, No print for debugging
DEBUG = DEBUG_LEVEL[0]

## test result 
LINE_OPTS = ["# text =", "# head_tail_text =", "# output =", ""] # "" means empty string
##TEST_LINE_OPT = "# output = "

##
HEAD_TAIL_VERIFICATION_TYPE = ["+", "!+"]

## Dictionary type 
##             0   ,     1    ,      2    ,     3     ,      4     ,               5            ,                 6
DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

def read_test_raw_corpus(path):
    """Reading data line by line from head-tail copurs from corpus directory

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    the corpus format is the following:

    # text = 잠실에서 우리가 더 강했다 .
    # head_tail_text = 잠실+에서 우리+가 더 강했+다 .
    # output = 잠실+에서 우리+가 더 강했+다 .

    # text = 이 정부가 , 국민에게 탄핵당한 정부가 왜 이렇게 사드 배치를 서두르는지 이해할 수가 없다 .
    # head_tail_text = 이 정부+가 , 국민+에게 탄핵당한 정부+가 왜 이렇+게 사드 배치+를 서두르+는지 이해할 수+가 없+다 .
    # output = 이 정부+가 , 국민+에게 탄핵당한 정부+가 왜 이렇+게 사드 배치+를 서두르+는지 이해할 수+가 없+다 .

    Arg:
      path(str): The path to a raw corpus to be read 
                 from test_result/#_kcc_000/output_file_name
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

        elif LINE_OPTS[2] in val: # output_text
            output_hetail = val 

            assert len(val.split()) > 3, "your data is empty, {}".format(val)

        elif LINE_OPTS[3] == val:
            temp_sent = sents.split()
            temp_hetail = hetail.split()
            temp_output_hetail = output_hetail.split()

            assert len(temp_sent) == len(temp_hetail), "len(temp_sent) != len(temp_hetail), temp_sent('{}'): {} and temp_hetail('{}'): {}".format(len(temp_sent), temp_sent, len(temp_hetail), temp_hetail)
            assert len(temp_hetail) == len(temp_output_hetail), "len(temp_hetail) != len(temp_output_hetail), temp_hetail('{}'): {} and temp_output_hetail('{}'): {}".format(len(temp_hetail), temp_hetail, len(temp_output_hetail), temp_output_hetail)
        else:  
            assert False, "read_raw_corpus function error"
      

    if DEBUG in DEBUG_LEVEL[0:]:
        print("\n===== Reading a file of {} =====".format(path))
        print("The number of lines: {}".format(len(data)))
        print("for top 5 of examples, \n{}".format(data[0:5]))

    return data


def remove_function(data):
    """head-tail pair like '검찰+이' split into '검찰' and '이'

    
    Args:
        data(list): list of tokens like ground_truth head-tail tokens and output

    Return:
        answer(list): if the token is with "+", split it 
                      if not, maintain 
                      final format is list 
    """

    _output = []

    for idx, val in enumerate(data):
        if "+" in val:
            _output.extend(val.split("+"))
        else:
            _output.append(val)


    ##if DEBUG in DEBUG_LEVEL[0:]:
    ##    print("\n===== remove_function =====")
    ##    print("The input data: {}\n{}".format(len(data), data))
    ##    print("The split data: {}\n{}".format(len(_output), _output))
    ##    input()

    return _output 


def measure_test(test_result_file_path, head_tail_types="None"):
    """measure precision and recall on the result of head-tail tokenizer

    using the structure of directory 
 
    TEST_RESULT_ROOT = "test_result/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    # For Debugging option, increasing level enumerate log in detail
    DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

    ## if DEBUG_LEVEL is 0, No print for debugging
    DEBUG = DEBUG_LEVEL[0]

    ## train and test corpus 
    LINE_OPTS = ["# text =", "# head_tail_text =", "# output =", ""] # "" means empty string
    ##TEST_LINE_OPT = "# output = "

    ## Dictionary type 
    ##             0   ,     1    ,      2    ,     3     ,      4     ,               5            ,                 6
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

    DICT_HEAD =  ["N-gram_word","Heat-tail_Token","Fre.","Prob."]

    ##
    HEAD_TAIL_VERIFICATION_TYPE = ["+", "!+"]
 
    Args: 
       test_result_file_path(str): the location of test reuslt
    Return: 
       None 
    """

    assert head_tail_types in HEAD_TAIL_VERIFICATION_TYPE, "head_tail_types is wrong, head_tail_types: {}".format(head_tail_type)
     
    assert isinstance(test_result_file_path, str), "The test_result_file_path is wrong in 'test' function: the type of path is {}, the value is {}".format(type(test_result_file_path), test_result_file_path)

    print("\n===== Preprocessing the file {} =====".format(test_result_file_path))

    raw_data = read_test_raw_corpus(test_result_file_path)

    print("\n===== Reading the file {} is done ! =====".format(test_result_file_path))

    ## train and test corpus 
    ## test result 
    ##LINE_OPTS = ["# text =", "# head_tail_text =", "# output =", ""] # "" means empty string

    ### calculate precision and recall
    ## the total cases with "# head_tail_text =" line
    total_for_recall = 0
    ## the total cases with "# output =" line
    total_for_precision = 0
    correct_num = 0

    print("\n====== measure scores with recall, precision, and f1 score with head_tail_types('{}') =====".format(head_tail_types))

    for idx, val in enumerate(raw_data):
        if LINE_OPTS[0] in val:

            temp_sent = val.split()[3:]

            ##print("\n\nInput Sentence: {}".format(temp_sent))

        elif LINE_OPTS[1] in val:

            pre_ground_truth_toks = val.split()[3:]
 
            ##print("\n\nPrevious Ground Truth toks: {}\n{}".format(len(pre_ground_truth_toks), pre_ground_truth_toks))

            if head_tail_types == HEAD_TAIL_VERIFICATION_TYPE[1]: # not including '+', !+
                ground_truth_toks = remove_function(pre_ground_truth_toks)
            else: # including '+', +
                ground_truth_toks = pre_ground_truth_toks
 
            total_for_recall += len(ground_truth_toks)
            ##print("Ground Truth toks: {}\n{}".format(len(ground_truth_toks), ground_truth_toks))

        elif LINE_OPTS[2] in val:          

            pre_output_toks = val.split()[3:] 

            ##print("\n\nPrevious output toks before: {}\n{}".format(len(pre_output_toks), pre_output_toks))

            if head_tail_types == HEAD_TAIL_VERIFICATION_TYPE[1]: # not including '+', !+
                output_toks = remove_function(pre_output_toks)
            else: # including '+'
                output_toks = pre_output_toks

            total_for_precision += len(output_toks)
            ##print("Output toks: {}\n{}".format(len(output_toks), output_toks))
 
            ## Check if the length of ground_truth and output of our model is the same or not
            ## This have to think about it
            assert len(pre_ground_truth_toks) == len(pre_output_toks), "check if len(pre_ground_truth_toks) == len(pre_output_toks) or not, pre_ground_truth_toks('{}'){} and pre_output_toks('{}'){}".format(len(pre_ground_truth_toks), pre_ground_truth_toks, len(pre_output_toks), pre_output_toks)


        elif LINE_OPTS[3] == val:
 
            ##if head_tail_types == HEAD_TAIL_VERIFICATION_TYPE[1]: # not including '+', !+
            ## For Debugging 
            ##print("\n\nOutput_toks: {}".format(output_toks))
            ##print("Ground_truth_toks: {}".format(ground_truth_toks))
            
            for predicted_idx, predicted_val in enumerate(output_toks):
                if predicted_val in ground_truth_toks:
                    correct_num += 1
                    ## For Debugging
                    ##print("\n====== Before del ======")
                    ##print("The predicted: {}\nground_truth_toks: {}".format(predicted_val, ground_truth_toks))
                    del ground_truth_toks[ground_truth_toks.index(predicted_val)]
                    ## For Debugging
                    ##print("\n===== After del =====") 
                    ##print("The predicted: {}\nground_truth_toks: {}".format(predicted_val, ground_truth_toks))

                    ##input()
                if ground_truth_toks == []:
                    break
            
            ##input()
            ##else: # including '+', !+
            ##for predicted_idx, predicted_val in enumerate(output_toks):
            ##    if predicted_val == ground_truth_toks[predicted_idx]:
            ##        correct_num += 1
        else:
            assert False, "Your dict type is wrong, the type of dict is {}".format(val)

    #### The result is printed in here #####
    print("\n======= measur test result function is done ! ===========")
    print("\n======= writing test result file is done in {} ! ===========".format(test_result_file_path))
    print("The total of sentences: {}, the 1/4: {}".format(len(raw_data), len(raw_data)/4))
    print("The actual tokens number(for recall): {}".format(total_for_recall))
    print("The predicted tokens number(for precision): {}".format(total_for_precision))
    print("The correct predicted tokens number: {}".format(correct_num))
    recall_score = correct_num/total_for_recall
    precision_score = correct_num/total_for_precision
    numerator = 2 * recall_score*precision_score
    denominator = recall_score + precision_score
    f_1_score = numerator/denominator
    print()
    print("Recall(correct_num/total_for_recall): {}".format(recall_score))
    print("Precision(correct_num/total_for_precision): {}".format(precision_score))
    print("F1 Score(2*recall*precision/(recall+precision)): {}".format(f_1_score))

 
if __name__ == "__main__":
    """
    TEST_RESULT_ROOT = "test_result/"

    CHILD_DIR = ["00_kcc_q28_only/", "01_kcc_150_only/", "02_kcc_150_n_q28/"]

    # For Debugging option, increasing level enumerate log in detail
    DEBUG_LEVEL = ["DEBUG_LEVEL_0", "DEBUG_LEVEL_1", "DEBUG_LEVEL_2"]

    ## if DEBUG_LEVEL is 0, No print for debugging
    DEBUG = DEBUG_LEVEL[0]

    ## train and test corpus 
    LINE_OPTS = ["# text =", "# head_tail_text =", "# output =", ""] # "" means empty string
    ##TEST_LINE_OPT = "# output = "

    ## Dictionary type 
    ##             0   ,     1    ,      2    ,     3     ,      4     ,               5            ,                 6
    DICT_TYPES = ["UNI", "LEFT_BI", "LEFT_TRI", "RIGHT_BI", "RIGHT_TRI", "BIDIRECTIONAL_BI_WINDOW_1", "BIDIRECTIONAL_TRI_WINDOW_2"]

    DICT_HEAD =  ["N-gram_word","Heat-tail_Token","Fre.","Prob."]

    ##
    HEAD_TAIL_VERIFICATION_TYPE = ["+", "!+"]
    """

    dict_types = DICT_TYPES[0] #

    child_dict = CHILD_DIR[0] #

    head_tail_tok_type = HEAD_TAIL_VERIFICATION_TYPE[0]

    ## For Test
    test_result_file_loc = TEST_RESULT_ROOT+child_dict[:-1]+"_with_"+dict_types+"_Dict_result.txt" #child_dict+child_dict[:-1]+"_with_"+dict_types+"_Dict_result.txt"

    ## For Experiments
   
    ###test_result_file_loc = TEST_RESULT_ROOT+child_dict+child_dict[:-1]+"_with_"+dict_types+"_Dict_result.txt"

    measure_test(test_result_file_loc, head_tail_tok_type)
