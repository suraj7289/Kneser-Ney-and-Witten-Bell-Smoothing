Steps to run my model - 

1. Copy the corpus.txt and Language_Model.py at the same folder
2. Run CMD from same folder, so we go to same drive
3. Run command -  python language_model.py <value of n> <smoothing type> <filename>

This will run for ngram count between 1 & 3.


Common implementing steps for Kneser Ney Smoothing & witten Bell - 

 1. Remove Punctuations, convert corpus to lower gram
 2. Word tokenize
 3. Create defaulldicts of ngrams(1,2,3)
 4. Create Kneser Ney and Witten Bell LM

Difference in their implementation Steps - 
 1. For calculating lowergram's Kneser Ney probability, we use discounted counts (C-kn), unlike witten bell.
 2. We use absolute discounting for highergrams, by substracting d (0.75 or 0.5 for ngrams) from ngram counts, unlike witten bell


Comparing Model's performance (based on probabilty of given input sentence) - 

1. When input sentence is combination of all unknown words - 
   For lower grams LM, witten bell gives better probability for a given sentence as compared to Kneser Ney, but when we move 
   towards higher ngram LM (ngram=3 in our case), Kneser Ney gives better probabilty.

2. When input sentence is combination of all known words (sentence taken from corpus) - 
   Kneser Ney gives better probability for lower grams LM, but as we move towards higher grams (ngram=3 in our case), 
   witten bell gives better probabilty of input sentence.

3. When Input sentence is combination of known and unknown words - 
   For ngrams = 1, witten bell gives better probability for a given sentence as compared to Kneser Ney, but when we move 
   towards higher ngram LM (ngram=2 onwards in our case), Kneser Ney gives better probabilty.
   

This observation helps us to know that as soon as we go to highergram witten bell model, model gives good probability of a sentence.
Overall Kneser Ney gives me better probabilty for ngram = 3 than Witten Bell.