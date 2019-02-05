#!/bin/bash

set -eu
#source ~/env.sh
echo "HERE " > /tmp/res.out
workdir=$1; shift
action=$1; shift
pidfile=$1; shift

function failsafe {
    set +eu
    $@
    set -eu
}

if [[ "$action" == "run" ]]; then
    log=$1; shift
    binary=$1; shift
    echo "GOGOGO >> $binary $@ " >> /tmp/res.out
    setsid $binary $@ > $log 2>&1 &
    echo $! > $pidfile
elif [[ "$action" == "wait" ]]; then
    pid=$(cat $pidfile)
    while kill -0 $pid; do
      sleep 0.5
    done

elif [[ "$action" == "kill" ]]; then
    pid=$(cat $pidfile)
    failsafe kill -- -$pid
    failsafe kill -9 -- -$pid
fi;

echo "DONE" >> /tmp/res.out
