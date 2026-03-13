## 1.BUILD TOKENIZER
### UTF-8
Load file with 'rb' or 'wb', because this assighment needs utf-8
### Special Tokens
Take good care of the 'special token'("\<endoftext>"). I just split by special tokens and call 're.finditer' to each splitted parts of the bytes.
### Lexicographically Greater Pair
I implement this by 

max_pair,max_num = max(pairs_counts.items(), key=lambda x: (x[1], x[0]))
### Optimize BPE
In my implement, I first iterate through a dict of {pretoken:counts} to find all adjcant pairs and get the dict of {pairs:counts} and {pretoken:[pairs]}. Then merge the maxpair and pop it from {pairs:counts}. Then search {pretoken:[pairs]} and deal with the last merged pairs front and behind pairs iteractively.