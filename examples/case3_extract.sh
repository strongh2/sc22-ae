#!/bin/bash


CMD1="grep -R 'SamplesPerSec' ./results/*zero*l-32*hs-2048*.txt | awk -v FS='[/,_: =]' '{print \$5, \$6, \$4,  \$22, \$23}' | sort"
CMD2="grep -R 'SamplesPerSec' ./results/*l-32*hs-2048*.txt --exclude=*zero*.txt | awk -v FS='[/,_: ]' '{print \$5, \$6, \$4, \$20, \$22}' | sort"

#echo -e 'Running: ' $CMD1 '\n'
#echo -e 'Running: ' $CMD2 '\n'

eval $CMD1 > ./results/case3.csv
eval $CMD2 >> ./results/case3.csv
	
