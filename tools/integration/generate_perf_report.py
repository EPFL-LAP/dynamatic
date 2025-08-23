def parse_results():
    return [
        {"name": "test1aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaathis is a very long name",
            "cycles": 1234, "result": "pass"},
        {"name": "test2", "result": "fail"},
        {"name": "test3", "result": "timeout"}
    ]


def table(header, data):
    res = ""
    for elem in header:
        res += f"| {elem} "
    res += "|\n"

    for elem in header:
        res += "| ---- "
    res += "|\n"

    for row in data:
        for elem in header:
            if elem in row:
                res += f"| {row[elem]} "
            else:
                res += f"| N/A "
        res += "|\n"

    return res


def main():
    print(table(
        ["name", "cycles", "result"],
        parse_results()
    ))


if __name__ == "__main__":
    main()
