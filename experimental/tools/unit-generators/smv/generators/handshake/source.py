def generate_source(name, params):
    return f"""
MODULE {name}(outs_ready)

    // output
    DEFINE outs_valid  := TRUE;
"""


if __name__ == "__main__":
    print(generate_source("test_source", {}))
