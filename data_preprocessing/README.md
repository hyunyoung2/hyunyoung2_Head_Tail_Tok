## Data preprocessing

This directory manages data directory. 

split total of data into train and test 

and then remove duplication of sentence. 

mkdir train and test corpus to ../corpus directory 


If you want to make head-tail corpus with pos tag

run this code line:

```
## With pos tagging
extract_function_call(pos_tag=True)
```

If you want not to make head-tail corpus with pos tag

run this code line:

```
## without pos tagging
## extract_function_call(pos_tag=False)
```
