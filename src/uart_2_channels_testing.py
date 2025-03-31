import serial
import csv
import os
from datetime import datetime

# Configuration
COM_PORT = '/dev/tty.usbmodem2103'
BAUD_RATE = 115200
FLUSH_INTERVAL = 10  # Flush every 10 samples

def main():
    # Resolve the path to ../Data_Gtec/adc_data.csv relative to this file
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, '..', 'Data_Gtec'))
    os.makedirs(data_dir, exist_ok=True)  # Create folder if it doesn't exist
    output_file = os.path.join(data_dir, 'adc_data.csv')

    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {COM_PORT} at {BAUD_RATE} baud.")
    except serial.SerialException as e:
        print(f"Failed to open serial port: {e}")
        return

    sample_count = 0

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ADC1', 'ADC2', 'ADC3', 'ADC4', 'ADC5', 'ADC6', 'Counter', 'Timestamp'])  # Updated header

        try:
            while True:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    parts = line.split(',')
                    if len(parts) == 7:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        sample_count += 1
                        writer.writerow([parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], sample_count, timestamp])
                        print(f"{sample_count}: {line}")

                        if sample_count % FLUSH_INTERVAL == 0:
                            f.flush()
        except KeyboardInterrupt:
            print("\nData collection stopped by user.")
        finally:
            ser.close()
            print("Serial port closed.")

if __name__ == '__main__':
    main()