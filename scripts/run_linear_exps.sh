#!/bin/bash

if [ -z "$1" ]
then
    echo "No argument supplied"
    exit 1
fi

if [ -z "$2" ]
then
    echo "No input dim argument supplied"
    exit 1
fi

if [ "$1" = "linear_ars" ]
then
    ./scripts/run_linear_ars.sh $2
elif [ "$1" = "linear_sgd" ]
then
    ./scripts/run_linear_sgd.sh $2
elif [ "$1" = "linear_reinforce" ]
then
    ./scripts/run_linear_reinforce.sh $2
elif [ "$1" = "linear_naturalreinforce" ]
then
    ./scripts/run_linear_naturalreinforce.sh $2
elif [ "$1" = "linear_newton" ]
then
    ./scripts/run_linear_newton.sh $2
else
    echo "Use one of {linear_ars, linear_sgd, linear_reinforce, linear_naturalreinforce, linear_newton} as arguments to the script"
fi
