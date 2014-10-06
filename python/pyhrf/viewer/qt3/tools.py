# -*- coding: utf-8 -*-
def cartesian(*sequences):
    """
    Generate the `cartesian product' of all SEQUENCES.  Each member of the
    product is a list containing an element taken from each original sequence.
    """
    length = len(sequences)
    #print 'sequences (%d):' %length
    #print sequences
    if length < 4:
        # Cases 1, 2 and 3 are for speed only, these are not really required.
        if length == 3:
            for first in sequences[0]:
                for second in sequences[1]:
                    for third in sequences[2]:
                        yield [first, second, third]
        elif length == 2:
            for first in sequences[0]:
                for second in sequences[1]:
                    yield [first, second]
        elif length == 1:
            for first in sequences[0]:
                yield [first]
        else:
            yield []
    else:
        head, tail = sequences[:-1], sequences[-1]
        #print 'head :', head
        #print 'tail :', tail
        for result in cartesian(*head):
            #print 'r :', result
            for last in tail:
                #if type(tail) == list :
                #    print '-> r+t:', result + tail
                #    yield result + tail
                #else:
                #print '-> r+t:', result + [last]
                yield result + [last]
