#!/bin/bash

for i in $(find . -type f \( -iname \*.cpp -o -iname \*.hpp \)); do
  if ! grep -q "Copyright" "$i"; then
    echo "$i"; cat "Copyright.txt" "$i" > "$i".new && mv "$i".new "$i"
  fi
done