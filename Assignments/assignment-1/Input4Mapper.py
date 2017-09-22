#!/usr/bin/env python

import sys
from string import digits

for l in sys.stdin:
    docId,key = l.strip().split('\t',1)
    print '%s\t%s' % (docId,key)