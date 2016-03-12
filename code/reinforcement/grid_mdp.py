'''
Created on Mar 13, 2016

@author: trucvietle
'''

import sys
import getopt

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg
    
def main(argv=None):
    ## Parse command line options
    if argv is None:
        argv = sys.argv
    ## Define the set of actions
    actions = ['N', 'S', 'E', 'W']
    print actions
    
    try:
        try:
            opts, args = getopt.getopt(argv[1:], 'h', ['help'])
        except getopt.error, msg:
            raise Usage(msg)
    except Usage, err:
        print >> sys.stderr, err.msg
        print >> sys.stderr, 'for help use --help'
        return 2
    
if __name__ == '__main__':
    sys.exit(main())
    