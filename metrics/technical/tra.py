import os
from os.path import join, split
import subprocess


def tra(input_dir, num_digits):
    if os.name == 'nt':
        script_root = join(
            os.path.dirname(__file__), "third_party", "Win", "TRAMeasure.exe"
        )
    elif os.name == 'posix':
        script_root = join(
            os.path.dirname(__file__), "third_party", "Linux", "TRAMeasure")
    else:
        raise NotImplementedError

    if "01" in split(input_dir)[-1]:
        sequence = "01"
    elif "02" in split(input_dir)[-1]:
        sequence = "02"
    else:
        raise ValueError
    dataset_dir = join(*split(input_dir)[:-1])

    command = f"{script_root} " \
              f"{dataset_dir} " \
              f"{sequence} {num_digits}"
    try:
        out = subprocess.check_output(command, shell=True)
        tra_measure = float(
            out.decode().replace("\n", "").replace("TRA measure: ", ""))
    except Exception as e:
        print(
            "Error while processing TRA measure with",
            command, e
        )
        tra_measure = None

    return tra_measure
