#!/usr/bin/env python

import sys

for line in sys.stdin:
    word,key = line.strip().split('\t',1)
    print '%s\t%s' % (word,key)
    