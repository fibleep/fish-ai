import serial
import time
import subprocess
import json
from typing import Optional, Dict, List

def find_arduino_port(auto_select: bool = True) -> Optional[str]:
    """Find Arduino port with option for automatic or manual selection."""
    print("Checking for all devices using PlatformIO...")
    try:
        result = subprocess.run(['pio', 'device', 'list', '--json-output'], 
                              capture_output=True, text=True, check=True)
        devices: List[Dict] = json.loads(result.stdout)
        
        if not devices:
            print("No devices found.")
            return None
            
        print(f"\nFound {len(devices)} device(s):")
        for i, device in enumerate(devices, 1):
            print(f"{i}. Port: {device.get('port')}, Description: {device.get('description', 'No description')}")
        
        if auto_select:
            selected_device = devices[-1]  # Select the last device
            port = selected_device.get('port')
            print(f"\nAutomatically selected device: {port}")
            print(f"Description: {selected_device.get('description', 'No description')}")
            
            confirmation = input("\nUse this device? (Y/n): ").strip().lower()
            if confirmation == 'n':
                return manual_select_device(devices)
            return port
        else:
            return manual_select_device(devices)
            
    except subprocess.CalledProcessError:
        print("Error: PlatformIO command failed. Make sure PlatformIO is installed.")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to parse device list output.")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def manual_select_device(devices: List[Dict]) -> Optional[str]:
    """Handle manual device selection."""
    while True:
        choice = input("\nEnter the number of the device to use, or 'q' to quit: ")
        if choice.lower() == 'q':
            return None
        try:
            choice = int(choice)
            if 1 <= choice <= len(devices):
                selected_device = devices[choice - 1]
                port = selected_device.get('port')
                print(f"Selected device: {port}")
                return port
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

def send_command(ser: serial.Serial, command: str) -> str:
    """Send command to Arduino and get response."""
    try:
        ser.write(f"{command}\n".encode())
        time.sleep(0.1)
        response = ser.readline().decode('ascii', errors='ignore').strip()
        print(f"Arduino response: {response}")
        return response
    except serial.SerialException as e:
        print(f"Error sending command: {e}")
        return ""

def main():
    # Find Arduino port with automatic selection
    port = find_arduino_port(auto_select=True)
    if port is None:
        print("No Arduino selected. Exiting.")
        return
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)  # Updated to match Arduino baud rate
        print("Waiting for Arduino to initialize...")
        time.sleep(2)  # Wait for the serial connection to initialize
    except serial.SerialException as e:
        print(f"Error opening serial port {port}: {e}")
        return

    print("\nArduino Control Started")
    print("Commands:")
    print("  1-3: Toggle motors")
    print("  'motor,speed': Set motor speed (e.g., '1,255')")
    print("  'q': Quit")
    
    try:
        while True:
            command = input("\nEnter command: ").strip().lower()
            
            if command == 'q':
                break
            
            if ',' in command:  # New speed control format
                parts = command.split(',')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    motor_num = int(parts[0])
                    speed = int(parts[1])
                    if 1 <= motor_num <= 3 and 0 <= speed <= 255:
                        send_command(ser, command)
                    else:
                        print("Invalid values. Motor: 1-3, Speed: 0-255")
                else:
                    print("Invalid format. Use 'motor,speed' (e.g., '1,255')")
            elif command in ['1', '2', '3']:
                send_command(ser, command)
            else:
                print("Invalid command. Use '1-3' or 'motor,speed' format.")
                
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        print("Closing serial connection...")
        ser.close()
        print("Arduino Control Ended")

if __name__ == "__main__":
    main()
