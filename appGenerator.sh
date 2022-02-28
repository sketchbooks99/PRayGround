#!bin/sh

if [ $# -ne 1 ]; then
    echo "Please configure the application name"
    exit 1
fi

if [ -e $1 ]; then
    echo "Directory $1 exists."
    exit
fi

mkdir -p $1

cp ./apps/empty_app/* $1

echo "$1 is created."