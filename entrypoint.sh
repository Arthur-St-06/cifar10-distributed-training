#!/bin/bash
set -e

# Start SSH server so MPI can connect
service ssh start

# Let mpirun call this script with python
exec "$@"
