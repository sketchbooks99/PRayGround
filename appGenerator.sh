#!bin/sh

# This is a very instant shell script to generate an application.

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

APP_NAME=$(basename $1)

# Replace the app name in CMakeLists.txt
sed -i "s/empty_app/$APP_NAME/g" $1/CMakeLists.txt

echo "$1 is created."