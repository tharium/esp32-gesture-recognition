import numpy as np
import csv

def parse_line(line):
    try:
        parts = line.split(',')
        csi_start = 5 # this is set to 5 to ignore the RSSI value and some other initial parts

        # Find end index (label)
        csi_end = -1
        for i, part in enumerate(parts):
            if part.strip(" '[]\"\n") in ('wave', 'control', 'clap', 'push', 'pull'): #gesture labels
                csi_end = i
                label = part.strip(" '[]\"\n")
                break
        if csi_end == -1:
            return None

        # Iterate CSI values
        csi_values = []
        for val in parts[csi_start:csi_end]:
            val = val.strip(" '[]\"\n")
            try:
                csi_values.append(int(val))
            except ValueError:
                continue

        return csi_values, label
    except:
        return None


def load_dataset(file_path):
    X = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # Each row is a list of strings
            line = ','.join(row)  # convert back to one string line
            parsed = parse_line(line)
            if parsed:
                features, label = parsed
                X.append(features)
                #y.append(1 if label == 'wave' else 0)  *** ground truth labels are set up in test file now
    return X