#!/usr/bin/env python

import sys

for line in sys.stdin:
    word,id_label = line.strip().split('\t',1)
    print '%s\t%s' % (word, id_label)