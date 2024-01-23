# Sample sequence of commands for Dynamatic frontend

# Indicate the path to Dynamatic's top-level directory here (leave unchanged if
# running the frontend from the top-level directory)
set-dynamatic-path  .

# Indicate the path the legacy Dynamatic's top-level directory here (required
# for write-hdl and simulate)
set-legacy-path     ../dynamatic-utils/legacy-dynamatic/dhls/etc/dynamatic

set-src             integration-test/bicg/bicg.c
compile             
write-hdl
simulate

# Exit the frontend
exit
