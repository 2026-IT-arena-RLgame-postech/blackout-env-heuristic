#!/bin/bash
# Run a lane after another lane's pid file disappears (one lane per GPU: chain the next lane onto the same GPU):
#   after_lane.sh PIDFILE lane.sh-args...
F=$1; shift
while [ -f "$F" ]; do sleep 60; done
exec bash "$(dirname "$0")/lane.sh" "$@"
