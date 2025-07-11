import json
import subprocess
import time
from typing import Dict, List, Optional

import serial


def find_arduino_port(auto_select: bool = True) -> Optional[str]:
    print("Checking for all devices using PlatformIO...")
    result = subprocess.run(
        ["pio", "device", "list", "--json-output"],
        capture_output=True,
        text=True,
        check=True,
    )
    devices: List[Dict] = json.loads(result.stdout)

    if not devices:
        print("No devices found.")
        return None

    print(f"\nFound {len(devices)} device(s):")
    for i, device in enumerate(devices, 1):
        print(
            f"{i}. Port: {device.get('port')}, Description: {device.get('description', 'No description')}"
        )

    if auto_select:
        selected_device = devices[-1]
        port = selected_device.get("port")
        print(f"\nAutomatically selected device: {port}")
        print(f"Description: {selected_device.get('description', 'No description')}")

        confirmation = input("\nUse this device? (Y/n): ").strip().lower()
        if confirmation == "n":
            return manual_select_device(devices)
        return port
    else:
        return manual_select_device(devices)


def manual_select_device(devices: List[Dict]) -> Optional[str]:
    while True:
        choice = input("\nEnter the number of the device to use, or 'q' to quit: ")
        if choice.lower() == "q":
            return None
        choice = int(choice)
        if 1 <= choice <= len(devices):
            selected_device = devices[choice - 1]
            port = selected_device.get("port")
            print(f"Selected device: {port}")
            return port
        else:
            print("Invalid choice. Please try again.")


def send_command(ser, command: str) -> str:
    ser.write(f"{command}\n".encode())
    time.sleep(0.1)
    response = ser.readline().decode("ascii", errors="ignore").strip()
    print(f"Arduino response: {response}")
    return response


def main():
    port = find_arduino_port(auto_select=True)
    if port is None:
        print("No Arduino selected. Exiting.")
        return

    ser = serial.Serial(port, 115200, timeout=1)
    print("Waiting for Arduino to initialize...")
    time.sleep(2)

    print("\nArduino Control Started")
    print("Commands:")
    print(
        "  'motor,speed': Set motor speed from -255 to 255 (e.g., '1,200' or '2,-150')"
    )
    print(
        "  'motor,speed,duration': Optionally include duration in ms (e.g., '1,255,1000')"
    )
    print("  'q': Quit")

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command == "q":
            break

        if "," in command:
            parts = command.split(",")
            try:
                motor_num = int(parts[0])
                speed = int(parts[1])
                if 1 <= motor_num <= 3 and -255 <= speed <= 255:
                    send_command(ser, command)
                else:
                    print("Invalid values. Motor: 1-3, Speed: -255 to 255.")
            except (ValueError, IndexError):
                print("Invalid command format. Use 'motor,speed' (e.g., '1,200')")
        else:
            print("Invalid command. Use 'motor,speed' or 'q'.")

    print("Closing serial connection...")
    ser.close()
    print("Arduino Control Ended")


if __name__ == "__main__":
    main()
