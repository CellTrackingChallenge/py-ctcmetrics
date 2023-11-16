

def print_results(results):
    """ Prints the results in a nice table.

    Args:
        results: A dictionary containing the results.
    """

    def print_line(metrics: dict):
        """ Prints a line of the table.

        Args:
            args: A list containing the arguments for the line.
        """

        print(*[f"{k}: {'N/A' if v is None else float(v):.5},\t" for k, v
                in metrics.items()])

    if type(results) is dict:
        print_line(results)
    elif type(results) is list:
        for res in results:
            print(res[0], end=":\t\t")
            print_line(res[1])


def store_results(path, results, delimiter=";"):
    """ Stores the results in a csv file.

    Args:
        path: The path to the csv file.
        results: A dictionary containing the results.
        delimiter: The delimiter for the csv file.
    """
    if not path.endswith(".csv"):
        path += ".csv"
    if type(results) is dict:
        keys = results.keys()
        with open(path, "w+") as f:
            f.write(delimiter.join(keys) + "\n")
            f.write(delimiter.join([str(v) for v in results.values()]) + "\n")
    elif type(results) is list:
        keys = results[0][1].keys()
        with open(path, "w+") as f:
            f.write('dataset'+delimiter+delimiter.join(keys) + "\n")
            for dataset, res in results:
                f.write(dataset+delimiter)
                f.write(delimiter.join([str(v) for v in res.values()]) + "\n")
