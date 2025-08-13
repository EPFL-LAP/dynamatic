import argparse

parser = argparse.ArgumentParser(
    description="Update frequencies.csv to the post-handshake-transformation CFG")
parser.add_argument(
    "--frequencies", type=str, required=True,
    help="Path to the frequencies.csv file")
parser.add_argument(
    "--mapping", type=str, required=True,
    help="Path to the mapping.csv file")

args = parser.parse_args()

# Construct mapping
mapping = {}
with open(args.mapping, 'r') as f:
    for line in f.readlines():
        key, value = line.strip().split(',')
        mapping[key] = value

with open(args.frequencies, 'r') as f_old:
    print(f_old.readline().strip())  # Write header
    for line in f_old.readlines():
        values = line.strip().split(',')
        # Update BBs
        # srcBlock
        values[0] = mapping[values[0]]
        # dstBlock
        values[1] = mapping[values[1]]
        # if srcBlock == dstBlock and not is_backedge, ignore the line
        if values[0] == values[1] and values[3] == "0":
            continue
        print(','.join(values))
