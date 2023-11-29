#!/bin/bash

# Check if a commit message was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <commit-message>"
    exit 1
fi

# Assign the first argument as the commit message
COMMIT_MESSAGE="$1"

# Execute the Git commands
git pull
git add .
git commit -m "$COMMIT_MESSAGE"
git push
