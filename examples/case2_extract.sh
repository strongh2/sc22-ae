#!/bin/bash

CMD1="grep -R 'SamplesPerSec' ./results/*zero*.txt | awk -v FS='[/,_: =]' '{print \$5, \$6, \$4,  \$22, \$23}' | sort"
CMD2="grep -R 'SamplesPerSec' ./results/*.txt --exclude=*zero*.txt | awk -v FS='[/,_: ]' '{print \$5, \$6, \$4, \$20, \$22}' | sort"

#echo -e 'Running: ' $CMD1 '\n'
#echo -e 'Running: ' $CMD2 '\n'

eval $CMD1 > ./results/case2.csv
eval $CMD2 >> ./results/case2.csv

CMD3="grep -R 'total parameters' ./results/*.txt  | awk -v FS='[/,_: =]' '{print \$5, \$6, \$4, \$20, \$21}' | uniq | sort"

eval $CMD3 >> ./results/case2.csv
