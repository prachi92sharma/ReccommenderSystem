#!/usr/bin/env python

import sys

for line in sys.stdin:
    docId,key = line.strip().split('\t',1)
    print '%s\t%s' % (docId,key)