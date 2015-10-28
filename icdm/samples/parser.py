#! /bin/env python
import re
import sys

# Look for bag | tuple | plain field
p = re.compile( r'(\{[\w()\.,-]+\})|(\([\w\.,-]+\))|([\w\.-]+)' )

def parseBag( tmps ):
    # some sanity check if tmps is a bag
    if tmps is None:
        return None
    s = tmps.strip()
    if len(s) == 0 or s[0] != '{' or s[-1] != '}':
        return None

    s = s.strip('{}')
    if len(s) == 0 :
        return []
    elif s[0] != '(' or s[-1] != ')':
        return None

    s = s.strip('()')
    return [ tuple(el.split(',')) for el in s.split('),(') if el != ''  and el is not None ]

def parseTuple( tmps ):
    if tmps is None:
        return None
    # some sanity check if tmps is a tuple
    s = tmps.strip()
    if len(s) == 0 or s[0] != '(' or s[-1] != ')':
        return None

    s = s.strip('()')
    if len(s) != 0 :
        return  tuple(s.split(','))

def parseLine( line ):
    s = line.strip('\n')
    l = []
    for m in p.finditer( s ):
        tup = m.groups()
        if len(tup) != 3:
            continue

        if tup[0] is not None:
        # a bag
            l.append( parseBag( tup[0] ) )
        elif tup[1] is not None:
        # a tuple
            l.append( parseTuple( tup[1] ) )
        else:
        # a plain field
            l.append( tup[2] )
    return l

for l in open(sys.argv[1]):
    print parseLine( l )
