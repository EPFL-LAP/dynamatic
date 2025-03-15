
#!/usr/bin/env bash

# Download the dot2smv archive from github
wget https://github.com/Jiahui17/dot2smv/archive/refs/heads/elastic-miter.tar.gz -O dot2smv.tar.gz
tar -zxf dot2smv.tar.gz
rm dot2smv.tar.gz

# Move the directory to ext and rename it to dot2smv
mkdir ext
mv dot2smv-elastic-miter ext/dot2smv
