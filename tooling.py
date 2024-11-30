import serial
import time
import subprocess
import json
from typing import Optional, Dict, List, Tuple
import re
from queue import Queue
import threading

class Tooling:
    def __init__(self):
        """Initialize the Tooling class with Arduino connection"""
        self.serial_conn = None
        self.port = None
        self.connect()
        self.mouth_state = "closed"
        self.command_queue = Queue()
        self.is_action_pending = False  # Flag for pending actions
        self.last_action_time = 0  # Track when the last action occurred
        self.command_thread = threading.Thread(target=self._process_command_queue, daemon=True)
        self.command_thread.start()

    def connect(self) -> bool:
        """Establish connection with Arduino"""
        self.port = self.find_arduino_port()
        if not self.port:
            print("Failed to find Arduino port")
            return False
        
        try:
            self.serial_conn = serial.Serial(self.port, 115200, timeout=1)
            time.sleep(2)  # Wait for Arduino initialization
            print(f"Successfully connected to Arduino on port {self.port}")
            return True
        except serial.SerialException as e:
            print(f"Failed to connect to Arduino: {e}")
            return False

    def find_arduino_port(self) -> Optional[str]:
        """Automatically find and connect to Arduino port"""
        try:
            result = subprocess.run(['pio', 'device', 'list', '--json-output'], 
                                  capture_output=True, text=True, check=True)
            devices: List[Dict] = json.loads(result.stdout)
            
            if not devices:
                print("No devices found")
                return None
                
            selected_device = devices[-1]
            return selected_device.get('port')
            
        except Exception as e:
            print(f"Error finding Arduino port: {e}")
            return None

    def _process_command_queue(self):
        """Process commands in the background"""
        while True:
            try:
                motor, speed, duration, is_action = self.command_queue.get()
                if motor is None:  # Shutdown signal
                    break
                
                if is_action:
                    self.is_action_pending = True
                    # Reset all motors before action
                    for m in range(1, 4):
                        if m != motor:  # Don't reset the motor we're about to use
                            self.serial_conn.write(f"{m},0\n".encode())
                            time.sleep(0.05)  # Small delay between resets
                
                command = f"{motor},{speed}"
                if duration is not None:
                    command += f",{int(duration * 1000)}"  # Convert to milliseconds
                command += "\n"
                
                if self.serial_conn:
                    print(f"Sending command: {command.strip()}")
                    self.serial_conn.write(command.encode())
                    
                    if duration:
                        time.sleep(duration)
                        # Only reset is_action_pending after duration
                        if is_action:
                            self.is_action_pending = False
                            self.last_action_time = time.time()
                    else:
                        if is_action:
                            self.is_action_pending = False
                            self.last_action_time = time.time()
                
            except Exception as e:
                print(f"Error processing command: {e}")
                self.is_action_pending = False

    def send_command(self, motor: int, speed: int, duration: Optional[float] = None, is_action: bool = False) -> bool:
        """
        Send command to Arduino
        Args:
            motor: Motor number (1-3)
            speed: Speed (0-255)
            duration: Optional duration in seconds
            is_action: Whether this is an action command (not mouth movement)
        """
        try:
            self.command_queue.put((motor, speed, duration, is_action))
            return True
        except Exception as e:
            print(f"Error queuing command: {e}")
            return False

    def can_move_mouth(self) -> bool:
        """Check if it's safe to move the mouth"""
        if self.is_action_pending:
            return False
        # Don't move mouth for a short period after an action
        if time.time() - self.last_action_time < 0.4:
            return False
        return True

    def extract_tools(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Extract all tools within << >> from the text"""
        tool_positions = {}
        cleaned_text = text
        
        for tool in re.findall(r"<<(.*?)>>", text):
            tool_positions[tool] = text.find(f"<<{tool}>>")
            cleaned_text = cleaned_text.replace(f"<<{tool}>>", "")

        print(f"Extracted tools: {tool_positions}")
        print(f"Cleaned text: {cleaned_text}")
            
        return cleaned_text, tool_positions

    def run_tool(self, tool: str) -> bool:
        """Run the specified tool"""
        try:
            print(f"Executing tool: {tool}")
            
            if tool == "MouthOpen":
                if self.can_move_mouth() and self.mouth_state == "closed":
                    self.send_command(1, 255, is_action=False)
                    self.mouth_state = "open"
                    return True
                
            elif tool == "MouthClose":
                if self.can_move_mouth() and self.mouth_state == "open":
                    self.send_command(1, 0, is_action=False)
                    self.mouth_state = "closed"
                    return True
                
            elif tool == "TailFlop":
                # Quick burst for tail movement
                return self.send_command(3, 255, duration=0.2, is_action=True)
                
            elif tool == "MoveHead&&Outward":
                return self.send_command(2, 255, is_action=True)
                
            elif tool == "MoveHead&&Inward":
                return self.send_command(2, 0, is_action=True)
                
            elif tool == "HeadFlop":
                # Quick burst for head movement
                return self.send_command(2, 255, duration=0.2, is_action=True)
            
            return False
            
        except Exception as e:
            print(f"Error running tool {tool}: {e}")
            return False

    def reset_state(self) -> bool:
        """Reset all motors to initial state"""
        try:
            print("Resetting all motors to initial state...")
            # Wait for any pending actions
            while self.is_action_pending:
                time.sleep(0.1)
            # Stop all motors
            for motor in range(1, 4):
                self.send_command(motor, 0, is_action=False)
            self.mouth_state = "closed"
            return True
        except Exception as e:
            print(f"Error resetting state: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Signal command thread to stop
            self.command_queue.put((None, None, None, None))
            self.command_thread.join(timeout=1)
            
            if self.serial_conn:
                # Stop all motors
                for motor in range(1, 4):
                    self.send_command(motor, 0, is_action=False)
                time.sleep(0.1)  # Small delay to ensure commands are sent
                self.serial_conn.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup()

if __name__ == "__main__":
    print("Initializing Tooling test sequence...")
    
    try:
        tools = Tooling()
        
        # Test sequence with proper delays
        sequence = [
            ("MouthOpen", 0.5),
            ("TailFlop", 0.5),
            ("HeadFlop", 0.5),
            ("MouthClose", 0.5),
            ("MoveHead&&Outward", 0.5),
            ("MoveHead&&Inward", 0.5)
        ]
        
        for action, delay in sequence:
            print(f"\nExecuting {action}...")
            tools.run_tool(action)
            time.sleep(delay)  # Wait between actions
            
    except Exception as e:
        print(f"Test sequence failed: {e}")
    finally:
        if 'tools' in locals():
            print("\nCleaning up...")
            tools.cleanup()
            print("Test sequence completed.")
