import asyncio
import os
import json
import httpx
from typing import List, Dict, Optional

class SimpleMCPWrapper:
    """Simple wrapper that calls Home Assistant REST API directly instead of MCP"""
    
    def __init__(self, ha_base_url: str = "http://smartlab.nerdlab.local:8123"):
        self.ha_base_url = ha_base_url
        self.api_token = os.getenv("API_ACCESS_TOKEN")
        self.tools = []
        self.openai_tools = []

    async def initialize(self):
        """Initialize with basic Home Assistant tools"""
        if not self.api_token:
            raise Exception("API_ACCESS_TOKEN not found in environment")
            
        print(f"Initializing simple HA connection to {self.ha_base_url}")
        
        # Test connection and get device info
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            response = await client.get(f"{self.ha_base_url}/api/", headers=headers, timeout=5.0)
            response.raise_for_status()
            
            # Get available devices/entities
            self.devices = await self._get_available_devices(client, headers)
            
        # Define basic tools that map to HA services
        self._define_basic_tools()
        print(f"Simple MCP wrapper initialized with {len(self.openai_tools)} tools.")

    async def _get_available_devices(self, client: httpx.AsyncClient, headers: Dict) -> Dict:
        """Get available devices from Home Assistant"""
        try:
            # Get all entities
            response = await client.get(f"{self.ha_base_url}/api/states", headers=headers, timeout=5.0)
            response.raise_for_status()
            entities = response.json()
            
            devices = {
                "lights": [],
                "switches": [],
                "areas": []
            }
            
            for entity in entities:
                entity_id = entity.get("entity_id", "")
                friendly_name = entity.get("attributes", {}).get("friendly_name", entity_id)
                
                if entity_id.startswith("light."):
                    devices["lights"].append({
                        "entity_id": entity_id,
                        "name": friendly_name,
                        "simple_name": entity_id.replace("light.", "").replace("_", " ").title()
                    })
                elif entity_id.startswith("switch."):
                    devices["switches"].append({
                        "entity_id": entity_id, 
                        "name": friendly_name,
                        "simple_name": entity_id.replace("switch.", "").replace("_", " ").title()
                    })
            
            print(f"Found {len(devices['lights'])} lights and {len(devices['switches'])} switches")
            return devices
            
        except Exception as e:
            print(f"Error getting devices: {e}")
            return {"lights": [], "switches": [], "areas": []}

    def get_device_list_for_prompt(self) -> str:
        """Get a formatted list of ALL devices for the system prompt"""
        if not hasattr(self, 'devices'):
            return ""
            
        device_text = "\n\nAvailable Devices:\n"
        
        if self.devices.get("lights"):
            device_text += f"Lights ({len(self.devices['lights'])} total):\n"
            light_names = [light["simple_name"] for light in self.devices["lights"]]
            # Group lights in rows of 5 for better readability
            for i in range(0, len(light_names), 5):
                row = light_names[i:i+5]
                device_text += f"  • {', '.join(row)}\n"
            
        if self.devices.get("switches"):
            device_text += f"Switches ({len(self.devices['switches'])} total):\n"
            switch_names = [switch["simple_name"] for switch in self.devices["switches"]]
            # Group switches in rows of 5 for better readability
            for i in range(0, len(switch_names), 5):
                row = switch_names[i:i+5]
                device_text += f"  • {', '.join(row)}\n"
        
        # Add special commands
        device_text += "\nSpecial Commands:\n"
        device_text += "  • Use 'all lights' to control all lights at once\n"
        device_text += "  • Use exact device names as shown above\n"
            
        return device_text

    def _define_basic_tools(self):
        """Define basic Home Assistant tools"""
        self.openai_tools = [
            {
                "name": "mcp_Home_Assistant_HassTurnOn",
                "description": "Turn on lights or other devices",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the device or area"},
                        "area": {"type": "string", "description": "Area name"},
                        "domain": {"type": "array", "items": {"type": "string"}, "description": "Device domains like 'light'"}
                    }
                }
            },
            {
                "name": "mcp_Home_Assistant_HassTurnOff", 
                "description": "Turn off lights or other devices",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the device or area"},
                        "area": {"type": "string", "description": "Area name"},
                        "domain": {"type": "array", "items": {"type": "string"}, "description": "Device domains like 'light'"}
                    }
                }
            },
            {
                "name": "mcp_Home_Assistant_HassLightSet",
                "description": "Set light brightness or color",
                "parameters": {
                    "type": "object", 
                    "properties": {
                        "name": {"type": "string", "description": "Light name"},
                        "brightness": {"type": "integer", "minimum": 0, "maximum": 100},
                        "color": {"type": "string", "description": "Color name"}
                    }
                }
            },
            {
                "name": "mcp_Home_Assistant_HassLightTurnOffAll",
                "description": "Turn off all lights in the home",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "mcp_Home_Assistant_HassLightTurnOnAll",
                "description": "Turn on all lights in the home",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

    def get_tools_for_llm(self) -> List[Dict]:
        return self.openai_tools
    
    def get_tool_names(self) -> List[str]:
        return [t["name"] for t in self.openai_tools]

    async def call_tool(self, name: str, args: Dict) -> Dict:
        """Call Home Assistant service directly via REST API"""
        try:
            print(f"Calling HA tool '{name}' with args {args}")
            
            if name == "mcp_Home_Assistant_HassTurnOn":
                # Check if domain is specified
                if "domain" in args and args["domain"]:
                    return await self._call_ha_service_for_domain(args["domain"][0], "turn_on", args)
                # For lights, use light domain; for others, use homeassistant domain
                elif self._is_light_device(args.get("name", "")):
                    return await self._call_ha_service("light", "turn_on", args)
                else:
                    return await self._call_ha_service("homeassistant", "turn_on", args)
            elif name == "mcp_Home_Assistant_HassTurnOff":
                # Check if domain is specified
                if "domain" in args and args["domain"]:
                    return await self._call_ha_service_for_domain(args["domain"][0], "turn_off", args)
                # For lights, use light domain; for others, use homeassistant domain  
                elif self._is_light_device(args.get("name", "")):
                    return await self._call_ha_service("light", "turn_off", args)
                else:
                    return await self._call_ha_service("homeassistant", "turn_off", args)
            elif name == "mcp_Home_Assistant_HassLightSet":
                return await self._call_ha_service("light", "turn_on", args)
            elif name == "mcp_Home_Assistant_HassLightTurnOffAll":
                return await self._turn_off_all_lights()
            elif name == "mcp_Home_Assistant_HassLightTurnOnAll":
                return await self._turn_on_all_lights()
            else:
                return {"error": f"Unknown tool: {name}"}
                
        except Exception as e:
            print(f"Error calling HA tool {name}: {e}")
            return {"error": str(e)}

    def _is_light_device(self, device_name: str) -> bool:
        """Check if a device name refers to a light"""
        if not device_name:
            return False
            
        device_name_lower = device_name.lower()
        
        # Check if it's explicitly a light
        if "light" in device_name_lower or "lamp" in device_name_lower:
            return True
            
        # Check if we have this device in our lights list
        if hasattr(self, 'devices') and self.devices.get('lights'):
            for light in self.devices['lights']:
                if (device_name_lower in light["name"].lower() or 
                    device_name_lower in light["simple_name"].lower()):
                    return True
        
        return False

    async def _call_ha_service(self, domain: str, service: str, args: Dict) -> Dict:
        """Call a Home Assistant service"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            
            # Map args to HA service data
            service_data = {}
            
            if "name" in args:
                # Try to resolve device name to entity_id
                entity_id = self._resolve_device_name(args["name"], domain)
                if entity_id:
                    service_data["entity_id"] = entity_id
                else:
                    # Fallback: construct entity_id from name
                    clean_name = args["name"].lower().replace(" ", "_")
                    service_data["entity_id"] = f"{domain}.{clean_name}"
                    
            if "area" in args:
                service_data["area_id"] = args["area"]
            if "brightness" in args:
                service_data["brightness_pct"] = args["brightness"]
            if "color" in args:
                service_data["color_name"] = args["color"]
                
            url = f"{self.ha_base_url}/api/services/{domain}/{service}"
            print(f"Calling HA service: {url} with data: {service_data}")
            
            response = await client.post(url, headers=headers, json=service_data, timeout=5.0)
            response.raise_for_status()
            
            return {"result": f"Successfully called {domain}.{service} for {service_data.get('entity_id', 'unknown')}"}

    def _resolve_device_name(self, name: str, preferred_domain: str = None) -> str:
        """Resolve a device name to its entity_id"""
        if not hasattr(self, 'devices'):
            return None
            
        name_lower = name.lower()
        
        # If it's already a valid entity_id format, return it directly
        if name_lower.startswith(f"{preferred_domain}.") and preferred_domain:
            print(f"Using entity_id directly: {name_lower}")
            return name_lower
        
        # Check lights first if no domain specified or if domain is light
        if not preferred_domain or preferred_domain == "light":
            for light in self.devices.get("lights", []):
                # Exact match on simple name
                if name_lower == light["simple_name"].lower():
                    print(f"Resolved '{name}' to light entity (exact): {light['entity_id']}")
                    return light["entity_id"]
                # Partial match in name or friendly name
                if (name_lower in light["name"].lower() or 
                    name_lower in light["simple_name"].lower()):
                    print(f"Resolved '{name}' to light entity (partial): {light['entity_id']}")
                    return light["entity_id"]
                # Handle entity_id without domain prefix
                entity_name = light["entity_id"].replace("light.", "").replace("_", " ").lower()
                if name_lower == entity_name or name_lower in entity_name:
                    print(f"Resolved '{name}' to light entity (entity match): {light['entity_id']}")
                    return light["entity_id"]
        
        # Check switches
        if not preferred_domain or preferred_domain == "switch":
            for switch in self.devices.get("switches", []):
                # Exact match on simple name
                if name_lower == switch["simple_name"].lower():
                    print(f"Resolved '{name}' to switch entity (exact): {switch['entity_id']}")
                    return switch["entity_id"]
                # Partial match in name or friendly name
                if (name_lower in switch["name"].lower() or 
                    name_lower in switch["simple_name"].lower()):
                    print(f"Resolved '{name}' to switch entity (partial): {switch['entity_id']}")
                    return switch["entity_id"]
                # Handle entity_id without domain prefix
                entity_name = switch["entity_id"].replace("switch.", "").replace("_", " ").lower()
                if name_lower == entity_name or name_lower in entity_name:
                    print(f"Resolved '{name}' to switch entity (entity match): {switch['entity_id']}")
                    return switch["entity_id"]
        
        print(f"Could not resolve device name '{name}' to entity_id")
        return None

    async def _call_ha_service_for_domain(self, domain: str, service: str, args: Dict) -> Dict:
        """Call a Home Assistant service for all entities in a domain"""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {self.api_token}"}
            
            # Get all entity IDs for the domain
            entity_ids = []
            if domain == "light" and hasattr(self, 'devices'):
                entity_ids = [light["entity_id"] for light in self.devices.get("lights", [])]
            elif domain == "switch" and hasattr(self, 'devices'):
                entity_ids = [switch["entity_id"] for switch in self.devices.get("switches", [])]
            
            if not entity_ids:
                return {"error": f"No entities found for domain {domain}"}
            
            service_data = {"entity_id": entity_ids}
            
            url = f"{self.ha_base_url}/api/services/{domain}/{service}"
            print(f"Calling HA service for domain: {url} with {len(entity_ids)} entities")
            
            response = await client.post(url, headers=headers, json=service_data, timeout=5.0)
            response.raise_for_status()
            
            return {"result": f"Successfully called {domain}.{service} for {len(entity_ids)} entities"}

    async def _turn_off_all_lights(self) -> Dict:
        """Turn off all lights in the home"""
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.api_token}"}
                
                # Get all light entity IDs
                entity_ids = []
                if hasattr(self, 'devices') and self.devices.get('lights'):
                    entity_ids = [light["entity_id"] for light in self.devices["lights"]]
                
                if not entity_ids:
                    return {"error": "No lights found"}
                
                print(f"Turning off all {len(entity_ids)} lights in batches...")
                
                # Process lights in batches of 5 to avoid timeout issues
                batch_size = 5
                successful_batches = 0
                failed_batches = 0
                
                for i in range(0, len(entity_ids), batch_size):
                    batch = entity_ids[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    print(f"Processing batch {batch_num}: {batch}")
                    
                    service_data = {"entity_id": batch}
                    url = f"{self.ha_base_url}/api/services/light/turn_off"
                    
                    try:
                        response = await client.post(url, headers=headers, json=service_data, timeout=5.0)
                        response.raise_for_status()
                        print(f"✅ Batch {batch_num} successful")
                        successful_batches += 1
                        
                        # Small delay between batches
                        if i + batch_size < len(entity_ids):
                            await asyncio.sleep(0.2)
                            
                    except Exception as batch_error:
                        print(f"❌ Batch {batch_num} failed: {batch_error}")
                        failed_batches += 1
                        continue
                
                if failed_batches == 0:
                    return {"result": f"Successfully turned off all {len(entity_ids)} lights in {successful_batches} batches"}
                else:
                    return {"result": f"Turned off lights: {successful_batches} batches succeeded, {failed_batches} failed"}
                    
        except Exception as e:
            print(f"Error in _turn_off_all_lights: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to turn off lights: {e}"}

    async def _turn_on_all_lights(self) -> Dict:
        """Turn on all lights in the home"""
        try:
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.api_token}"}
                
                # Get all light entity IDs
                entity_ids = []
                if hasattr(self, 'devices') and self.devices.get('lights'):
                    entity_ids = [light["entity_id"] for light in self.devices["lights"]]
                
                if not entity_ids:
                    return {"error": "No lights found"}
                
                print(f"Turning on all {len(entity_ids)} lights in batches...")
                
                # Process lights in batches of 5 to avoid timeout issues
                batch_size = 5
                successful_batches = 0
                failed_batches = 0
                
                for i in range(0, len(entity_ids), batch_size):
                    batch = entity_ids[i:i + batch_size]
                    batch_num = (i // batch_size) + 1
                    print(f"Processing batch {batch_num}: {batch}")
                    
                    service_data = {"entity_id": batch}
                    url = f"{self.ha_base_url}/api/services/light/turn_on"
                    
                    try:
                        response = await client.post(url, headers=headers, json=service_data, timeout=5.0)
                        response.raise_for_status()
                        print(f"✅ Batch {batch_num} successful")
                        successful_batches += 1
                        
                        # Small delay between batches
                        if i + batch_size < len(entity_ids):
                            await asyncio.sleep(0.2)
                            
                    except Exception as batch_error:
                        print(f"❌ Batch {batch_num} failed: {batch_error}")
                        failed_batches += 1
                        continue
                
                if failed_batches == 0:
                    return {"result": f"Successfully turned on all {len(entity_ids)} lights in {successful_batches} batches"}
                else:
                    return {"result": f"Turned on lights: {successful_batches} batches succeeded, {failed_batches} failed"}
                    
        except Exception as e:
            print(f"Error in _turn_on_all_lights: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to turn on lights: {e}"}

    async def close(self):
        """No cleanup needed for simple wrapper"""
        pass