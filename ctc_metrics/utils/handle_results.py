

def print_results(results: dict):
    """
    Prints the results in a nice table.

    Args:
        results: A dictionary containing the results.
    """

    def print_line(metrics: dict):
        """
        Prints a line of the table.

        Args:
            metrics: A list containing the arguments for the line.
        """

        print(*[f"{k}: {'N/A' if v is None else float(v):.5},\t" for k, v
                in metrics.items()])

    if isinstance(results, dict):
        print_line(results)
    elif isinstance(results, list):
        for res in results:
            print(res[0], end=":\t\t")
            print_line(res[1])


def store_results(
        path: str,
        results: dict,
        delimiter: str = ";"):
    """
    Stores the results in a csv file.

    Args:
        path: The path to the csv file.
        results: A dictionary containing the results.
        delimiter: The delimiter for the csv file.
    """
    if not path.endswith(".csv"):
        path += ".csv"
    if isinstance(results, dict):
        keys = results.keys()
        with open(path, "w+", encoding="utf-8") as f:
            f.write(delimiter.join(keys) + "\n")
            f.write(delimiter.join([str(v) for v in results.values()]) + "\n")
    elif isinstance(results, list):
        keys = results[0][1].keys()
        with open(path, "w+", encoding="utf-8") as f:
            f.write('dataset'+delimiter+delimiter.join(keys) + "\n")
            for dataset, res in results:
                f.write(dataset+delimiter)
                f.write(delimiter.join([str(v) for v in res.values()]) + "\n")
