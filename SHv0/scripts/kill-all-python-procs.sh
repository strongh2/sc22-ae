cmd1="ps -ef | grep python | grep -v grep | grep -v '/etc/dsw/bin/python' | awk '{print $2}'"
cmd2="ps -ef | grep 'ray::' | grep -v grep | awk '{print $2}'"

while true; do

    u=$(eval $cmd1 | grep -v 'kill-all-python-procs' | wc | awk '{print $1}')
    #u=$(eval $cmd1 | wc | awk '{print $1}')
    if [ $u -ne 0 ]; then
    	eval $cmd1 | grep -v 'kill-all-python-procs' | awk '{print $2}' | xargs kill -9
	continue
    fi

    u=$(eval $cmd2 | grep -v 'kill-all-python-procs' | wc | awk '{print $1}')
    if [ $u -ne 0 ]; then
    	eval $cmd2 | grep -v 'kill-all-python-procs' | awk '{print $2}' | xargs kill -9
	continue
    fi

    break
done
#./stop-mps.sh
#sleep 15

#./init-mps.sh
#sleep 15
