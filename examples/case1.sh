#!/bin/bash

CMD="grep -R 'total parameters' ./results/*.txt | awk -v FS='[/,_: ]' '{print \$4, \$18, \$19, \$20, \$21}' | uniq"

echo -e 'Running: ' $CMD '\n'
eval $CMD
