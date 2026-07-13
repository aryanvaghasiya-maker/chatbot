from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
class AgentDisplay:
    def __init__(self):
        self.console = Console()
        self.printed_count = 0
        self.spinner = Spinner("dots", text="Deep Agent is thinking...", style="bold magenta")
    def print_message(self, msg, is_final_turn: bool = False):
        if isinstance(msg, HumanMessage):
            self.console.print(f"\n[bold green]You 💬:[/bold green] {msg.content}")
        elif isinstance(msg, AIMessage):
            # Print tool calls if any
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "unknown")
                    args = tc.get("args", {})
                    self.console.print(f"[dim yellow]⚙️ Running Tool: {name}({args})[/dim yellow]")
            
            # Extract and clean text content
            content = msg.content
            if isinstance(content, list):
                text_parts = []
                for p in content:
                    if isinstance(p, dict) and p.get("type") == "text":
                        text_parts.append(p.get("text", ""))
                    elif isinstance(p, str):
                        text_parts.append(p)
                content = "\n".join(text_parts)
                
            if is_final_turn:
                if content.strip():
                    self.console.print(Panel(content, title="🤖 Agent Response", border_style="blue"))
            else:
                if content.strip():
                    self.console.print(f"[bold blue]Agent 🤖:[/bold blue] {content}")
        elif isinstance(msg, ToolMessage):
            name = msg.name or "tool"
            self.console.print(f"[dim cyan]🔧 Tool Result [{name}]: {msg.content}[/dim cyan]")
        elif isinstance(msg, SystemMessage):
            self.console.print(f"[dim grey]System: {msg.content}[/dim grey]")
