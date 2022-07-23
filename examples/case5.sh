#!/bin/bash

CMD="grep -R 'elapsed time per iteration' ./results/*stronghold*l-32*hs-2048*.txt | awk -v FS='[/,_:|]' '{print \$8, \$4, \$12, \$13}' | uniq | sort"

echo -e 'Running: ' $CMD '\n'
eval $CMD
	
