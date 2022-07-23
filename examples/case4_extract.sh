#!/bin/bash

CMD="grep -R 'elapsed time per iteration' ./results/*stronghold*ws-15*.txt | awk -v FS='[/,_:|]' '{print \$5, \$6, \$8, \$4, \$12, \$13}' | uniq | sort"

#echo -e 'Running: ' $CMD '\n'
eval $CMD > ./results/case4.csv
	
