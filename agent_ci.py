import inspect
from typing import Any, Callable, Dict, List, Optional, TypeVar

# Define a type variable for tools, assuming they are callables or objects with a 'run' method
ToolType = TypeVar('ToolType', bound=Callable[..., Any])

class ToolRegistry:
    """
    A registry for managing and accessing various tools available to an agent.
    Tools can be any callable or object that the agent can invoke.
    """
    def __init__(self):
        self._tools: Dict[str, Any] = {}

    def register_tool(self, tool: Any, name: Optional[str] = None):
        """
        Registers a tool with the registry.

        Args:
            tool: The tool to register. This can be a function, a class instance,
                  or any callable object.
            name: An optional name for the tool. If not provided, the tool's
                  __name__ attribute (for functions) or __class__.__name__
                  (for objects) will be used.
        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        tool_name = name
        if tool_name is None:
            if hasattr(tool, '__name__'):
                tool_name = tool.__name__
            elif hasattr(tool, '__class__') and hasattr(tool.__class__, '__name__'):
                tool_name = tool.__class__.__name__
            else:
                raise ValueError("Could not determine a name for the tool. Please provide one.")

        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("Tool name must be a non-empty string.")

        if tool_name in self._tools:
            raise ValueError(f"Tool with name '{tool_name}' is already registered.")

        self._tools[tool_name] = tool
        # print(f"Tool '{tool_name}' registered successfully.")

    def get_tool(self, name: str) -> Optional[Any]:
        """
        Retrieves a registered tool by its name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The tool object if found, otherwise None.
        """
        return self._tools.get(name)

    def get_all_tools(self) -> Dict[str, Any]:
        """
        Returns a dictionary of all registered tools, mapping their names to the tool objects.

        Returns:
            A dictionary of all registered tools.
        """
        return self._tools.copy()

    def list_tool_names(self) -> List[str]:
        """
        Returns a list of names of all registered tools.

        Returns:
            A list of tool names.
        """
        return list(self._tools.keys())

    def unregister_tool(self, name: str) -> bool:
        """
        Unregisters a tool by its name.

        Args:
            name: The name of the tool to unregister.

        Returns:
            True if the tool was successfully unregistered, False otherwise.
        """
        if name in self._tools:
            del self._tools[name]
            # print(f"Tool '{name}' unregistered successfully.")
            return True
        # print(f"Tool '{name}' not found in registry.")
        return False

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __str__(self) -> str:
        return f"ToolRegistry with {len(self)} tools: {', '.join(self.list_tool_names())}"

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"

# Example usage (for testing purposes, not part of the final agent_ci.py)
# if __name__ == "__main__":
#     def search_web(query: str) -> str:
#         return f"Searching for: {query}"

#     class Calculator:
#         def add(self, a: int, b: int) -> int:
#             return a + b
#         def subtract(self, a: int, b: int) -> int:
#             return a - b

#     registry = ToolRegistry()

#     # Register a function
#     registry.register_tool(search_web)
#     print(registry)

#     # Register a class instance
#     calc_instance = Calculator()
#     registry.register_tool(calc_instance, name="calculator")
#     print(registry)

#     # Get and use tools
#     web_tool = registry.get_tool("search_web")
#     if web_tool:
#         print(web_tool("Python programming"))

#     calc_tool = registry.get_tool("calculator")
#     if calc_tool:
#         print(calc_tool.add(5, 3))
#         print(calc_tool.subtract(10, 4))

#     # Try to register with duplicate name
#     try:
#         registry.register_tool(search_web, name="search_web")
#     except ValueError as e:
#         print(f"Error: {e}")

#     # List all tools
#     print("All tools:", registry.get_all_tools())
#     print("Tool names:", registry.list_tool_names())

#     # Unregister a tool
#     registry.unregister_tool("search_web")
#     print(registry)
#     print("Tool names after unregister:", registry.list_tool_names())