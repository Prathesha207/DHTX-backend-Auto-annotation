import os
import json
import importlib.util
from pathlib import Path
from app.plugins.base_plugin import BaseMLPlugin

class PluginManager:
    def __init__(self, plugins_dir: str = "app/plugins"):
        self.plugins_dir = Path(__file__).resolve().parents[2] / plugins_dir
        self.loaded_plugins = {}

    def discover_plugins(self):
        plugins = []
        if not self.plugins_dir.exists():
            return plugins
            
        for plugin_path in self.plugins_dir.iterdir():
            if plugin_path.is_dir() and (plugin_path / "config.json").exists():
                with open(plugin_path / "config.json", "r") as f:
                    config = json.load(f)
                plugins.append({
                    "id": plugin_path.name,
                    "path": plugin_path,
                    "config": config
                })
        return plugins

    def load_plugin(self, plugin_id: str) -> BaseMLPlugin:
        if plugin_id in self.loaded_plugins:
            return self.loaded_plugins[plugin_id]
            
        plugin_dir = self.plugins_dir / plugin_id
        config_path = plugin_dir / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Plugin config not found: {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        entry_file = config.get("entry_file", "plugin.py")
        entry_class = config.get("entry_class", "Plugin")
        
        module_path = plugin_dir / entry_file
        if not module_path.exists():
            raise FileNotFoundError(f"Plugin entry file not found: {module_path}")
            
        spec = importlib.util.spec_from_file_location(f"plugins.{plugin_id}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        plugin_class = getattr(module, entry_class)
        plugin_instance = plugin_class()
        
        if not isinstance(plugin_instance, BaseMLPlugin):
            raise TypeError(f"Plugin {plugin_id} does not implement BaseMLPlugin")
            
        self.loaded_plugins[plugin_id] = plugin_instance
        return plugin_instance
