import serial
import time
import subprocess
import json
from typing import Optional, Dict, List, Tuple, Set
from dataclasses import dataclass
import re
import threading
from enum import Enum, auto

@dataclass
class MotorControl:
    speed: int = 0
    end_time: float = 0
    is_moving: bool = False
    last_command_time: float = 0
    queued_stop: bool = False

class Tooling:
    def __init__(self):
        self.serial_conn = None
        self.port = None
        
        self.motors = {
            1: MotorControl(),  
            2: MotorControl(), 
            3: MotorControl()  
        }
        
        self.cooldowns = {
            "TailFlop": 0.1,
            "HeadFlop": 0.1,
            "MoveHead&&Outward": 0.05,
            "MoveHead&&Inward": 0.05
        }
        self.last_actions = {action: 0 for action in self.cooldowns}
        
        self.mouth_open = False
        self.last_mouth_move = 0
        self.MOUTH_INTERVAL = 0.02
        
        self._init_connection()

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def _find_arduino_port(self) -> Optional[str]:
        result = subprocess.run(['pio', 'device', 'list', '--json-output'], 
                              capture_output=True, text=True, check=True)
        return json.loads(result.stdout)[-1].get('port')

    def _init_connection(self):
        self.port = self._find_arduino_port()
        if not self.port:
            raise Exception("Failed to find Arduino port")
        
        self.serial_conn = serial.Serial(self.port, 115200, timeout=1)
        time.sleep(2)
        print(f"Connected to Arduino on {self.port}")
        
        for motor in range(1, 4):
            self._send_immediate_command(motor, 0)
            time.sleep(0.1)

    def _update_loop(self):
        while self.running:
            current_time = time.time()
            
            for motor_id, motor in self.motors.items():
                if motor.is_moving and motor.end_time > 0 and current_time >= motor.end_time and not motor.queued_stop:
                    print(f"Stopping motor {motor_id} after duration")
                    self._send_immediate_command(motor_id, 0)
                    motor.is_moving = False
                    motor.end_time = 0
                    motor.queued_stop = True
            
            time.sleep(0.01)

    def _send_immediate_command(self, motor: int, speed: int, duration: Optional[float] = None):
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Serial connection lost")
            return
            
        cmd = f"{motor},{speed}"
        if duration:
            cmd += f",{int(duration * 1000)}"
        cmd += "\n"
        
        print(f"Sending command: {cmd.strip()}")
        self.serial_conn.write(cmd.encode())
        self.serial_conn.flush()
        
        motor_ctrl = self.motors[motor]
        motor_ctrl.speed = speed
        motor_ctrl.is_moving = speed != 0
        if duration:
            motor_ctrl.end_time = time.time() + (duration * 1.1)
        else:
            motor_ctrl.end_time = 0
        motor_ctrl.last_command_time = time.time()
        motor_ctrl.queued_stop = False
        
        time.sleep(0.01)

    def move_mouth(self, open_mouth: bool) -> bool:
        current_time = time.time()
        if current_time - self.last_mouth_move < self.MOUTH_INTERVAL:
            return False
            
        if open_mouth != self.mouth_open:
            self._send_immediate_command(1, 255 if open_mouth else 0)
            self.mouth_open = open_mouth
            self.last_mouth_move = current_time
            return True
        return False

    def run_tool(self, tool: str) -> bool:
        current_time = time.time()
        
        if tool.startswith("Mouth"):
            return self.move_mouth(tool == "MouthOpen")
        
        if tool in self.cooldowns:
            if current_time - self.last_actions[tool] < self.cooldowns[tool]:
                return False
            self.last_actions[tool] = current_time
        
        if tool == "TailFlop":
            print(f"Executing TailFlop at {time.time()}")
            self._send_immediate_command(3, 255, 0.3)
            return True
            
        elif tool == "HeadFlop":
            self._send_immediate_command(2, 255, 0.2)
            return True
            
        elif tool == "MoveHead&&Outward":
            self._send_immediate_command(2, 255)
            return True
            
        elif tool == "MoveHead&&Inward":
            self._send_immediate_command(2, 0)
            return True
            
        print(f"Unknown tool: {tool}")
        return False

    def extract_tools(self, text: str) -> Tuple[str, Dict[str, int]]:
        tools = {}
        clean_text = text
        
        for tool in re.findall(r"<<(.*?)>>", text):
            tools[tool] = text.find(f"<<{tool}>>")
            clean_text = clean_text.replace(f"<<{tool}>>", "")

        print(f"Extracted tools: {tools}")
        return clean_text, tools

    def reset_state(self) -> bool:
        time.sleep(0.1)
        
        for motor in range(1, 4):
            self._send_immediate_command(motor, 0)
        self.mouth_open = False
        return True

    def cleanup(self):
        print("Starting cleanup...")
        self.running = False
        
        if hasattr(self, 'update_thread'):
            for motor in self.motors.values():
                if motor.is_moving and motor.end_time > 0:
                    remaining_time = motor.end_time - time.time()
                    if remaining_time > 0:
                        time.sleep(remaining_time)
            
            self.update_thread.join(timeout=1)
        
        if self.serial_conn and self.serial_conn.is_open:
            self.reset_state()
            time.sleep(0.2)
            self.serial_conn.close()
            
        print("Cleanup complete")

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    tooling = Tooling()
    print("\nTesting basic commands...")
    
    test_sequence = [
        ("TailFlop", 0.5),
        ("HeadFlop", 0.3),
        ("MoveHead&&Outward", 0.3),
        ("MoveHead&&Inward", 0.3)
    ]
    
    for action, delay in test_sequence:
        print(f"\nExecuting {action}...")
        tooling.run_tool(action)
        time.sleep(delay)
    
    tooling.cleanup()
