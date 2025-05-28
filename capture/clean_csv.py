import csv   

label = "pull"
infile = label + '_log.csv'
outfile = 'cleaned_' + infile

def parse_line(line):
    try:
        parts = line.split(',')

        # RSSI (4th item)
        rssi = int(parts[3].strip())

        # CSI starts at index 25 if header is not modified in firmware
        csi_start = 25  

        # Find end index (label)
        csi_end = -1
        for i, part in enumerate(parts):
            if part.strip().strip('[]"\''" ") == label:
                csi_end = i - 1
                break
        if csi_end == -1:
            print(f"Skipping line due to error: '{label}' not found")
            return None

        # Extract CSI values
        csi_values = []
        for val in parts[csi_start:csi_end]:
            val = val.strip(' \'[]"\n')
            #csi_values.append(int(val))
            if val.lstrip('-').isdigit():
                csi_values.append(int(val))

        return [rssi] + csi_values + [label]

    except Exception as e:
        print(f"Skipping line due to error: {e}")
        return None

def clean_csi_csv(input_path, output_path, max_csi_len=128):
    with open(input_path, 'r', encoding='latin1') as infile, open(output_path, 'w', newline='', encoding='latin1') as outfile:
        writer = csv.writer(outfile)
        
        # Header: rssi, csi_1 .. csi_n, label
        header = ['rssi'] + [f'csi_{i}' for i in range(1, max_csi_len + 1)] + ['label']
        writer.writerow(header)

        for line in infile:
            parsed = parse_line(line)
            if parsed:
                # Pad with 0s if CSI is shorter than expected
                while len(parsed) < len(header):
                    parsed.insert(-1, 0)
                writer.writerow(parsed)


clean_csi_csv(infile, outfile)