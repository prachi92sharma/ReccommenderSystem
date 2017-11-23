#!/usr/bin/env python

import sys

for l in sys.stdin:
    word,key = l.strip().split('\t',1)
    print '%s\t%s' % (word,key)