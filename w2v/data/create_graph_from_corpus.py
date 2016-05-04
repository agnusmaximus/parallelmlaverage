from __future__ import print_function
import sys
from random import shuffle

if len(sys.argv) != 4:
    print("Usage: ./create_graph_from_corpus.py corpus output percentage")
    exit(0)

percentage = float(sys.argv[3])

DISTANCE = 10

with open(sys.argv[1], 'r') as corpus:
    text = corpus.read()
    text_length = len(text)
    text = text[:int(percentage*text_length)]

    words_list = list(set(text.split()))
    word_to_id = {}

    # Write word id mappings
    f = open("word_id_mappings", "w")
    for index, word in enumerate(list(set(words_list))):
        #print("%d %s" % (index, word), file=f)
        word_to_id[word] = index
    f.close()

    # Construct graph
    g = {}
    words = text.strip().split(" ")
    lines = [words[i:i+DISTANCE] for i in range(len(words))]
    for line in lines:
        if len(line) < DISTANCE:
            continue
        first_word = line[0]
        for other_word in line:
            if other_word == first_word:
                continue
            a, b = tuple(sorted([first_word, other_word]))
            if (a,b) not in g:
                g[(a, b)] = 0
            g[(a, b)] += 1

    # Output graph to file
    f = open(sys.argv[2], "w")
    for word_pair, occ in g.items():
        print("%d %d %d" % (word_to_id[word_pair[0]], word_to_id[word_pair[1]], occ), file=f)

    # Print stats
    print("N_NODES=%d" % (len(set(list(words_list)))+1))
    print("N_EDGES=%d" % len(g.items()))
