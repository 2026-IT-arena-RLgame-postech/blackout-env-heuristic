#!/bin/bash
# Run a lane after another lane's pid file disappears (keeps R3 at 4 concurrent lanes on gpu2):
#   after_lane.sh PIDFILE lane.sh-args...
F=$1; shift
while [ -f "$F" ]; do sleep 60; done
exec bash "$(dirname "$0")/lane.sh" "$@"
