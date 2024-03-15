#!/bin/sh
# collected from https://gist.github.com/hbsdev/a17deea814bc10197285
# recursively removes all .pyc files and __pycache__ directories in the current
# directory

find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf
