import serial
import csv

# Adjust COM port and baud rate as needed
ser = serial.Serial('COM4', 115200, timeout=1)
output_file = 'adc_data.csv'

# Define headers for 9 ADC channels (and an extra "Sample" column if desired)
headers = ['Sample', 'ADC1', 'ADC2', 'ADC3', 'ADC4', 'ADC5', 'ADC6', 'ADC7', 'ADC8', 'ADC9']

with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    sample_count = 0
    try:
        while True:
            # Read one line from the serial port, with error handling for decoding issues.
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception as decode_err:
                print("Decoding error:", decode_err)
                continue

            if line:
                print("Received:", line)
                parts = line.split(',')
                # Case 1: If the device sends only the 9 ADC values, add a sample counter.
                if len(parts) == 9:
                    sample_count += 1
                    writer.writerow([sample_count] + parts)
                # Case 2: If the device sends a sample number along with 9 channels.
                elif len(parts) == 10:
                    writer.writerow(parts)
                else:
                    print("Unexpected data format:", parts)
    except KeyboardInterrupt:
        print("Data collection stopped by user.")
    finally:
        ser.close()