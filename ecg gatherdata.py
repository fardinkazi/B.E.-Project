import serial
import csv
import time

# Serial port config
ser = serial.Serial('COM9', 115200, timeout=2)
time.sleep(2)

# Target range from dataset
MIN_VOLTAGE = -7.0903741
MAX_VOLTAGE = 7.4021031

# Sample settings
required_samples = 140
ecg_values = []

print(f"Collecting {required_samples} ECG samples every 0.3 seconds...")

try:
    while len(ecg_values) < required_samples:
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if "LEAD_OFF" in line or line == '':
            print("LEAD_OFF or empty. Waiting...")
            continue

        try:
            raw_adc = float(line)  # Value should be in 0–1023
            if 0 <= raw_adc <= 1023:
                # Scale raw ADC to voltage range
                scaled_voltage = (raw_adc / 1023.0) * (MAX_VOLTAGE - MIN_VOLTAGE) + MIN_VOLTAGE
                ecg_values.append(round(scaled_voltage, 6))
                print(f"{len(ecg_values)}: {scaled_voltage:.6f}")
                time.sleep(0.3)  # Wait 0.3 seconds before next sample
            else:
                print(f"Ignored out-of-range value: {raw_adc}")

        except ValueError:
            print(f"Skipped non-numeric line: {line}")

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    ser.close()
    print("Serial port closed.")

# Save to CSV
with open("ecg_input.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(ecg_values)

print(f"\n✅ {len(ecg_values)} scaled ECG samples saved to ecg_input.csv")
