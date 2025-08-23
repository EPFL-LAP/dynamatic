def parse_results():
    return [
        {"name": "test1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaathis is a very long name",
            "cycles": 1234, "result": "pass"},
        {"name": "test2", "cycles": -1, "result": "fail"},
        {"name": "test3", "cycles": -1, "result": "timeout"}
    ]


def main():
    print("## Performance Report")
    print()
    print("| Name | Result | Num. cycles |")
    print("| ---- | ------ | ----------- |")
    t = parse_results()
    for d in t:
        print(f"| {d["name"]} ", end="")
        if d["result"] == "pass":
            print(f"| {d["cycles"]} ", end="")
        else:
            print(f"| N/A ", end="")
        print(f"| {d["result"]} ", end="")
        print("|")


if __name__ == "__main__":
    main()
