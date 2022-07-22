#!/bin/bash

CMD1="grep -R 'SamplesPerSec' ./results/*zero*.txt | awk -v FS='[/,_: ]' '{print \$5, \$6, \$4,  \$22}' | sort"
CMD2="grep -R 'SamplesPerSec' ./results/*.txt --exclude=*zero*.txt | awk -v FS='[/,_: ]' '{print \$5, \$6, \$4, \$20, \$22}' | sort"

echo -e 'Running: ' $CMD1 '\n'
echo -e 'Running: ' $CMD2 '\n'

eval $CMD1
eval $CMD2
	
