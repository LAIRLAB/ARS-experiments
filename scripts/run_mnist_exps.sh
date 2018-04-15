#!/bin/bash

if [ -z "$1" ]
then
    echo "No argument supplied"
    exit 1
fi

if [ "$1" = "mnist_ars" ]
then
    ./scripts/run_mnist_ars.sh
elif [ "$1" = "mnist_sgd" ]
then
    ./scripts/run_mnist_sgd.sh
elif [ "$1" = "mnist_reinforce" ]
then
    ./scripts/run_mnist_reinforce.sh
else
    echo "Use one of {mnist_ars, mnist_sgd, mnist_reinforce} as arguments to the script"
fi
