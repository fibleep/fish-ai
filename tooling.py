import serial
import time
import subprocess
import json
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass
import re
from queue import PriorityQueue
import threading
from enum import Enum, auto

class MotorState(Enum):
    IDLE = auto()
    BUSY = auto()
    PENDING = auto()

@dataclass(order=True)
class ToolCommand:
    priority: int
    timestamp: float
    motor: int
    speed: int
    duration: Optional[float]
    is_action: bool
    tool_name: str = ''
    block_other_motors: bool = False

class Tooling:
    def __init__(self):
        """Initialize the Tooling class with Arduino connection"""
        self.serial_conn = None
        self.port = None
        
        # Motor states
        self.motor_states = {1: MotorState.IDLE, 2: MotorState.IDLE, 3: MotorState.IDLE}
        self.motor_locks = {
            1: threading.Lock(),  # Mouth motor
            2: threading.Lock(),  # Head motor
            3: threading.Lock()   # Tail motor
        }
        
        # Separate queues for each motor
        self.command_queues = {
            1: PriorityQueue(),  # Mouth motor
            2: PriorityQueue(),  # Head motor
            3: PriorityQueue()   # Tail motor
        }
        
        # Mouth specific state
        self.mouth_state = "closed"
        self.mouth_last_change = 0
        self.MIN_MOUTH_INTERVAL = 0.1  # Minimum time between mouth movements
        
        # Action cooldowns
        self.action_cooldowns = {
            "TailFlop": 0.5,
            "HeadFlop": 0.5,
            "MoveHead&&Outward": 0.3,
            "MoveHead&&Inward": 0.3
        }
        self.last_action_times = {action: 0 for action in self.action_cooldowns}
        
        # Initialize connection
        try:
            self._init_connection()
        except Exception as e:
            print(f"Failed to initialize connection: {e}")
            raise

        # Start command processing threads - one for each motor
        self.running = True
        self.command_threads = {
            motor: threading.Thread(target=self._process_motor_queue, args=(motor,), daemon=True)
            for motor in range(1, 4)
        }
        for thread in self.command_threads.values():
            thread.start()

    def _find_arduino_port(self) -> Optional[str]:
        """Find the Arduino port using platformio"""
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

    def _init_connection(self):
        """Initialize the serial connection to Arduino"""
        self.port = self._find_arduino_port()
        if not self.port:
            raise Exception("Failed to find Arduino port")
        
        try:
            self.serial_conn = serial.Serial(self.port, 115200, timeout=1)
            time.sleep(2)  # Wait for Arduino initialization
            print(f"Successfully connected to Arduino on port {self.port}")
        except serial.SerialException as e:
            raise Exception(f"Failed to connect to Arduino: {e}")

    def _can_execute_action(self, tool_name: str) -> bool:
        """Check if an action can be executed based on cooldowns"""
        if tool_name not in self.action_cooldowns:
            return True
            
        current_time = time.time()
        cooldown = self.action_cooldowns[tool_name]
        last_time = self.last_action_times[tool_name]
        
        return current_time - last_time >= cooldown

    def _process_motor_queue(self, motor: int):
        """Process commands for a specific motor"""
        while self.running:
            try:
                if self.command_queues[motor].empty():
                    time.sleep(0.01)
                    continue
                
                command = self.command_queues[motor].get()
                
                # Skip if cooldown hasn't elapsed for actions
                if command.is_action and not self._can_execute_action(command.tool_name):
                    self.command_queues[motor].put(command)  # Put it back in queue
                    time.sleep(0.01)
                    continue
                
                with self.motor_locks[motor]:
                    self.motor_states[motor] = MotorState.BUSY
                    
                    # Send command to Arduino
                    cmd_str = f"{motor},{command.speed}"
                    if command.duration is not None:
                        cmd_str += f",{int(command.duration * 1000)}"
                    cmd_str += "\n"
                    
                    if self.serial_conn:
                        print(f"Sending command: {cmd_str.strip()}")
                        self.serial_conn.write(cmd_str.encode())
                        
                        if command.duration:
                            time.sleep(command.duration)
                            # Reset motor to stopped state after duration
                            if command.is_action:
                                stop_cmd = f"{motor},0\n"
                                self.serial_conn.write(stop_cmd.encode())
                            
                        if command.is_action:
                            self.last_action_times[command.tool_name] = time.time()
                    
                    self.motor_states[motor] = MotorState.IDLE
                    
            except Exception as e:
                print(f"Error processing command for motor {motor}: {e}")
                self.motor_states[motor] = MotorState.IDLE

    def can_move_mouth(self) -> bool:
        """Check if it's safe to move the mouth"""
        current_time = time.time()
        if current_time - self.mouth_last_change < self.MIN_MOUTH_INTERVAL:
            return False
        if self.motor_states[1] != MotorState.IDLE:
            return False
        return True

    def queue_command(self, motor: int, speed: int, duration: Optional[float] = None, 
                     is_action: bool = False, priority: int = 2, tool_name: str = '',
                     block_other_motors: bool = False) -> bool:
        """Queue a command for a specific motor"""
        try:
            command = ToolCommand(
                priority=priority,
                timestamp=time.time(),
                motor=motor,
                speed=speed,
                duration=duration,
                is_action=is_action,
                tool_name=tool_name,
                block_other_motors=block_other_motors
            )
            self.command_queues[motor].put(command)
            return True
        except Exception as e:
            print(f"Error queuing command: {e}")
            return False

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
        """Run the specified tool with priority handling"""
        try:
            print(f"Executing tool: {tool}")
            
            if tool == "MouthOpen":
                if self.can_move_mouth() and self.mouth_state == "closed":
                    self.queue_command(1, 255, is_action=False, priority=1, tool_name=tool)
                    self.mouth_state = "open"
                    self.mouth_last_change = time.time()
                    return True
                
            elif tool == "MouthClose":
                if self.can_move_mouth() and self.mouth_state == "open":
                    self.queue_command(1, 0, is_action=False, priority=1, tool_name=tool)
                    self.mouth_state = "closed"
                    self.mouth_last_change = time.time()
                    return True
                
            elif tool == "TailFlop":
                # Tail flop is completely independent
                return self.queue_command(3, 255, duration=0.2, is_action=True, 
                                       priority=2, tool_name=tool)
                
            elif tool == "MoveHead&&Outward":
                return self.queue_command(2, 255, is_action=True, 
                                       priority=2, tool_name=tool)
                
            elif tool == "MoveHead&&Inward":
                return self.queue_command(2, 0, is_action=True, 
                                       priority=2, tool_name=tool)
                
            elif tool == "HeadFlop":
                return self.queue_command(2, 255, duration=0.2, is_action=True, 
                                       priority=2, tool_name=tool)
            
            return False
            
        except Exception as e:
            print(f"Error running tool {tool}: {e}")
            return False

    def reset_state(self) -> bool:
        """Reset all motors to initial state"""
        try:
            print("Resetting all motors to initial state...")
            # Queue reset commands with highest priority
            for motor in range(1, 4):
                self.queue_command(motor, 0, is_action=False, priority=1)
            self.mouth_state = "closed"
            return True
        except Exception as e:
            print(f"Error resetting state: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.running = False
            for thread in self.command_threads.values():
                thread.join(timeout=1)
            
            if self.serial_conn:
                # Stop all motors
                for motor in range(1, 4):
                    cmd = f"{motor},0\n"
                    self.serial_conn.write(cmd.encode())
                time.sleep(0.1)
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
            ("MouthOpen", 0.2),
            ("TailFlop", 0.2),
            ("HeadFlop", 0.2),
            ("MouthClose", 0.2),
            ("MoveHead&&Outward", 0.2),
            ("MoveHead&&Inward", 0.2)
        ]
        
        for action, delay in sequence:
            print(f"\nExecuting {action}...")
            tools.run_tool(action)
            time.sleep(delay)
            
    except Exception as e:
        print(f"Test sequence failed: {e}")
    finally:
        if 'tools' in locals():
            print("\nCleaning up...")
            tools.cleanup()
            print("Test sequence completed.")