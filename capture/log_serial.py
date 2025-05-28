import serial
import csv
import time

"""
Captures serial data from a device, set to COM3 at 115200 baud rate.
May not be the same for all devices and may need to be adjusted.
You will have to change the label variable for each different gesture.
Note: This could use refinement to avoid rerunning the entire script for each gesture
I just haven't got around to it yet.
"""

# Set up the serial connection
ser = serial.Serial('COM3', 115200)
logging = False
label = "pull"
data_buffer = []

input_file = label + '_log.csv'

def serial_read():
    global logging, data_buffer

    timeout_t = time.time() + 10  # 10 second timeout
    packet_count = 0

    while len(data_buffer) < 100:
        if time.time() > timeout_t:
            print("Timeout reached")
            break

        if logging:
            try:
                line = ser.readline().decode('latin-1').strip()
                if line.startswith("CSI_DATA"):
                    row = line.split(' ')
                    row.append(label)
                    data_buffer.append(row)
                    print(packet_count)
                    packet_count += 1
            except Exception as e:
                print(f"Error reading serial: {e}")

def main():
    global logging, data_buffer

    with open(input_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        input("Press ENTER to start logging (100 packets)...")

        print("Starting data collection...")
        logging = True

        serial_read()

        logging = False
        ser.close()

        confirm = input("Enter 'p' to save data, any other key to discard: ")
        if confirm.lower() == 'p':
            for row in data_buffer:
                if "Task" not in row:
                    csv_writer.writerow([row])
                else:
                    print("Skipped task line")
            print("Data saved to CSV.")
        else:
            print("Data discarded.")

        print("Program finished.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        ser.close()
        print("\nSerial closed by user.")