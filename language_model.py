#!/usr/bin/env python
# coding: utf-8

# In[340]:


from collections import *
import os
import re
import sys
import string
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict


# In[341]:


def prepare_ngram_token_types(corpus_name):
    corp = open(corpus_name,'r')
    corp = corp.read()
    corp = corp.lower()
    corp = re.sub(r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", corp)
    token = word_tokenize(corp)
    print("Total Token count :",len(token))

    trigrams = get_ngrams(corp, 3)
    trigram_type = list(set(trigrams))
    bigrams = get_ngrams(corp, 2)
    bigram_type = list(set(bigrams))
    unigrams = token
    unigram_type = list(set(unigrams))
    
    
    trigram_dict = defaultdict(int) 
    for gram in trigrams:
        trigram_dict[gram] += 1 
        
    bigram_dict = defaultdict(int) 
    for gram in bigrams:
        bigram_dict[gram] += 1 
    
    unigram_dict = defaultdict(int) 
    for gram in unigrams:
        unigram_dict[gram] += 1
        
    return trigrams,trigram_type,bigrams,bigram_type,unigrams,unigram_type,trigram_dict,bigram_dict,unigram_dict
        
def get_ngrams(corpus,n):
    token = word_tokenize(corpus)
    ngrams = zip(*[token[i:] for i in range(n)])
    ngram_list = [" ".join(gram) for gram in ngrams]
    return ngram_list


# In[342]:


def get_lambda_unigram_context(unigrams,unigram_type,d):
    lambda_unigram = (d*(len(unigram_type)))/len(unigrams)
    return lambda_unigram

def get_lambda_bigram_context(unigram_dict,bigram_type,bigram_context,d):
    bigram_types_starts_with_word = 0
    unigram_count = 1
        
    for gram in bigram_type:
        if gram.split()[0] == bigram_context:
            bigram_types_starts_with_word = bigram_types_starts_with_word + 1
    
    if unigram_dict[bigram_context] == 0:
        unigram_count = 1
    else:
        unigram_count = unigram_dict[bigram_context]
    
    lambda_bigram_context = (d*bigram_types_starts_with_word)/unigram_count
    return lambda_bigram_context
            

def get_lambda_trigram_context(bigram_dict,trigram_type,trigram_context,d):
    trigram_types_starts_with_bigramword = 0
    bigram_count = 1
    
    for gram in trigram_type:
        tokens = gram.split()
        first_2_word = " ".join(tokens[0:2])
        if first_2_word == trigram_context:
            trigram_types_starts_with_bigramword = trigram_types_starts_with_bigramword + 1
            
    if bigram_dict[trigram_context] == 0:
        bigram_count = 1
    else:
        bigram_count = bigram_dict[trigram_context]
        
    lambda_trigram_context = (d*trigram_types_starts_with_bigramword)/bigram_count
    return lambda_trigram_context


# In[343]:


def calculate_trigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_dict,trigram_type,trigram_word,d):
    word = " ".join(trigram_word.split()[-1:])
    context = " ".join(trigram_word.split()[0:-1])
    lowergram = " ".join(trigram_word.split()[1:])
    
    kn_prob_trigram_lowergram = calculate_bigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_type,lowergram,d,normal=1)
    lambda_trigram_context = get_lambda_trigram_context(bigram_dict,trigram_type,context,d)
    if lambda_trigram_context==0:
        lambda_trigram_context=1
    second_part_tri = lambda_trigram_context*kn_prob_trigram_lowergram
    if bigram_dict[context] == 0:
        first_part_tri = 0
    else:
        first_part_tri = (max((trigram_dict[trigram_word]-d),0))/(bigram_dict[context])
    return first_part_tri + second_part_tri
    
        
def calculate_bigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_type,bigram_word,d,normal):
    word = " ".join(bigram_word.split()[-1:])
    context = " ".join(bigram_word.split()[0:-1])
    c_kn_numerator = 0
    c_kn_den = 0
    
    lambda_bigram_context = get_lambda_bigram_context(unigram_dict,bigram_type,context,d)
    if lambda_bigram_context == 0:
        lambda_bigram_context = 1
    
    kn_prob_bigram_word = calculate_unigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,word,d,1)
    second_part_bi = lambda_bigram_context*kn_prob_bigram_word
    
    if normal ==0:
        if unigram_dict[context] == 0:
            first_part_bi = 0 
        else:
            first_part_bi = (max((bigram_dict[bigram_word]-d),0))/(unigram_dict[context])
    else:
        for g in trigram_type:
            last_2_word = " ".join(g.split()[1:])
            last_word = " ".join(g.split()[-1:])
            if last_2_word == bigram_word:
                c_kn_numerator = c_kn_numerator + 1
            if last_word == context:
                c_kn_den = c_kn_den + 1
        if c_kn_den == 0 :
            first_part_bi = 0
        else:
            first_part_bi = max((c_kn_numerator-d),0)/c_kn_den
    
    return first_part_bi+second_part_bi

    
    
def calculate_unigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,unigram_word,d,normal):
    c_kn_numerator = 0
    lambda_unigram_context = get_lambda_unigram_context(unigrams,unigram_type,d)   #this is lambda of empty string, which is context of any unigram
    second_part = lambda_unigram_context/(len(unigram_type))
    if normal==0:
        first_part = max((unigram_dict[unigram_word]-d),0)/(len(unigrams))
    else:
        for g in bigram_type:
            if g.split()[1] == unigram_word:
                c_kn_numerator = c_kn_numerator + 1
        first_part = max((c_kn_numerator-d),0)/(len(bigram_type))
    
    return first_part+second_part    


# In[344]:


def calculate_sentence_probability_kn(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_dict,trigram_type,sentence,ngram):
    sentence = sentence.lower()
    sentence = re.sub(r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", sentence)
    sent_length = len(word_tokenize(sentence))
    input_tokens = word_tokenize(sentence)
    if sent_length < ngram:
        ngram=sent_length
        
    sent_probability = 1
    if ngram==3:
        trigram_context_unigram = "".join(input_tokens[0])
        trigram_context_bigram = " ".join(input_tokens[0:2])
        sent_probability =  sent_probability * calculate_bigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_type,trigram_context_bigram,0.5,normal=0)
        sent_probability =  sent_probability * calculate_unigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,trigram_context_unigram,0.5,normal=0)
        trigram_list = get_ngrams(sentence, 3)
        for gram in trigram_list:
            kn_prob = calculate_trigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_dict,trigram_type,gram,0.5)
            sent_probability = sent_probability*kn_prob
    elif ngram==2:
        bigram_context_unigram = "".join(input_tokens[0])
        sent_probability =  sent_probability * calculate_unigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,bigram_context_unigram,0.5,normal=0)
        bigram_list = get_ngrams(sentence, 2)
        for gram in bigram_list:
            kn_prob = calculate_bigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_type,gram,0.5,normal=0)
            sent_probability = sent_probability*kn_prob
    elif ngram==1:
        unigram_list = get_ngrams(sentence, 1)
        for gram in unigram_list:
            kn_prob = calculate_unigram_kn_prob(unigrams,unigram_type,unigram_dict,bigram_type,gram,0.5,normal=0)
            sent_probability = sent_probability*kn_prob
    print("Kneser Ney probability of sentence is ",sent_probability)


# In[345]:


def calc_unigram_wb_prob(unigrams,unigram_type,unigram_dict,unigram_word):
    unigram_word_count = unigram_dict[unigram_word]
    lambda_empty = 1 - (len(unigram_type)/(len(unigram_type) + len(unigrams)))
    wb_prob_unigram = (lambda_empty * unigram_word_count)/len(unigrams) + (1-lambda_empty)/len(unigram_type)
    return wb_prob_unigram

def calc_bigram_wb_prob(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,bigram_word):
    word = " ".join(bigram_word.split()[-1:])
    context = " ".join(bigram_word.split()[0:-1])
    bigram_word_count = bigram_dict[bigram_word]
    
    lambda_wb_bigram_context = calc_lambda_wb_context(unigram_dict,bigram_dict,trigram_dict,context,2)
    if unigram_dict[context] == 0:
        unigram_dict_context = 1
    else:
         unigram_dict_context = unigram_dict[context]
    first_part = lambda_wb_bigram_context* (bigram_word_count/unigram_dict_context)
    second_part = (1-lambda_wb_bigram_context)*calc_unigram_wb_prob(unigrams,unigram_type,unigram_dict,word)
    return first_part + second_part

def calc_trigram_wb_prob(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,trigram_word):
    word = " ".join(trigram_word.split()[-1:])
    context = " ".join(trigram_word.split()[0:-1])
    lowergram = " ".join(trigram_word.split()[1:])
    
    trigram_word_count = trigram_dict[trigram_word]
    
    lambda_wb_trigram_context = calc_lambda_wb_context(unigram_dict,bigram_dict,trigram_dict,context,3)
    
    if bigram_dict[context] == 0:
        bigram_dict_context = 1
    else:
        bigram_dict_context = bigram_dict[context]
        
    first_part = lambda_wb_trigram_context* (trigram_word_count/bigram_dict_context)
    second_part = (1-lambda_wb_trigram_context)*calc_bigram_wb_prob(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,lowergram)
    return first_part + second_part


def calc_lambda_wb_context(unigram_dict,bigram_dict,trigram_dict,wb_context,n):
    ngram_types_starts_with_context_wb = 0
    ngram_starts_with_context_wb = 0
    
    if n==2:
        dict1 = bigram_dict
    elif n==3:
        dict1 = trigram_dict
        
    for gram in dict1:
        tokens = gram.split()
        first_n_word_wb = " ".join(tokens[0:n-1])
        if first_n_word_wb == wb_context:
            ngram_types_starts_with_context_wb = ngram_types_starts_with_context_wb + 1
            ngram_starts_with_context_wb = ngram_starts_with_context_wb + dict1[gram]
    
    if ngram_starts_with_context_wb == 0:
        ngram_starts_with_context_wb = 1
        
    lambda_wb_ngram_context = 1 - (ngram_types_starts_with_context_wb/(ngram_types_starts_with_context_wb + ngram_starts_with_context_wb))
    return lambda_wb_ngram_context


# In[346]:


def calculate_sentence_probability_wb(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,sentence,ngram):
    sentence = sentence.lower()
    sentence = re.sub(r"[!#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", " ", sentence)
    sent_length = len(word_tokenize(sentence))
    input_tokens = word_tokenize(sentence)
    if sent_length < ngram:
        ngram=sent_length
        
    sent_probability = 1
    if ngram==3:
        trigram_context_unigram = "".join(input_tokens[0])
        trigram_context_bigram = " ".join(input_tokens[0:2])
        sent_probability =  sent_probability * calc_bigram_wb_prob(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,trigram_context_bigram)
        sent_probability =  sent_probability * calc_unigram_wb_prob(unigrams,unigram_type,unigram_dict,trigram_context_unigram)
        trigram_list = get_ngrams(sentence, 3)
        for gram in trigram_list:
            wb_prob = calc_trigram_wb_prob(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,gram)
            sent_probability = sent_probability*wb_prob
    elif ngram==2:
        bigram_context_unigram = "".join(input_tokens[0])
        sent_probability =  sent_probability * calc_unigram_wb_prob(unigrams,unigram_type,unigram_dict,bigram_context_unigram)
        bigram_list = get_ngrams(sentence, 2)
        for gram in bigram_list:
            wb_prob = calc_bigram_wb_prob(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,gram)
            sent_probability = sent_probability*wb_prob
    elif ngram==1:
        unigram_list = get_ngrams(sentence, 1)
        for gram in unigram_list:
            wb_prob = calc_unigram_wb_prob(unigrams,unigram_type,unigram_dict,gram)
            sent_probability = sent_probability*wb_prob
    print("Witten Bell probability of sentence is ",sent_probability)


# In[349]:


def main():
    arguments = sys.argv[1:]
    print(arguments)
    n = int(arguments[0])
    smoothing_type = arguments[1]
    fileName = arguments[2]
    if n <1 or n>3:
        print('Please put ngrams between 1 and 3')
        sys.exit()
    if smoothing_type =='w' or smoothing_type =='k':
        pass
    else:
        print('Please enter k for KneserNey or w for Witten Bell.')
        sys.exit()
        
    print("Corpus name is : ",fileName)
    trigrams,trigram_type,bigrams,bigram_type,unigrams,unigram_type,trigram_dict,bigram_dict,unigram_dict = prepare_ngram_token_types(fileName)
    if smoothing_type =='k':
        sentence =input('Please enter a sentence to calculate Probability : ')
        calculate_sentence_probability_kn(unigrams,unigram_type,unigram_dict,bigram_type,bigram_dict,trigram_dict,trigram_type,sentence,2)

    if smoothing_type =='w':
        sentence =input('Please enter a sentence to calculate Probability : ')
        calculate_sentence_probability_wb(unigrams,unigram_type,unigram_dict,bigram_dict,trigram_dict,sentence,2)


# In[350]:


if __name__ == "__main__":
    main()


# In[ ]:




