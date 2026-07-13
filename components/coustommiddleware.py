from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable, Awaitable

class StatefulCacheMiddleware(AgentMiddleware):
    """Intercepts agent workflows, establishing temporary virtual configurations."""
    def __init__(self, session_id: str):
        self.session_id = session_id

    def _modify_request(self, request: ModelRequest) -> ModelRequest:
        from langchain_core.messages import SystemMessage
        sys_msg = request.system_message
        addition = (
            f"\n\n[VOLATILE CACHE ACTIVE]\n"
            f"Session ID: {self.session_id}\n"
            f"You are inside an interactive chat loop. Every time you draft, write, edit, or generate any final content "
            f"(including blog posts, LinkedIn posts, tweets, or articles), you MUST save it as a file using your filesystem tools (e.g. `write_file` or `write`).\n"
            f"Do NOT just output the final draft in your text response. Always save it to the staging cache "
            f"(e.g. `linkedin/<slug>/post.md` for LinkedIn, `blogs/<slug>.md` for blogs) so the user can preview it in their dashboard.\n"
            f"Explicitly tell the user the path where you saved/staged the file.\n"
            f"If you generate an image, save it alongside the post using `generate_social_image`.\n"
            f"[/VOLATILE CACHE]"
        )
        if sys_msg is None:
            sys_msg = SystemMessage(content=addition)
        else:
            if isinstance(sys_msg.content, str):
                new_content = sys_msg.content + addition
            elif isinstance(sys_msg.content, list):
                new_content = list(sys_msg.content)
                if new_content and isinstance(new_content[-1], dict) and new_content[-1].get("type") == "text":
                    last_block = dict(new_content[-1])
                    last_block["text"] = last_block.get("text", "") + addition
                    new_content[-1] = last_block
                elif new_content and isinstance(new_content[-1], str):
                    new_content[-1] = new_content[-1] + addition
                else:
                    new_content.append({"type": "text", "text": addition})
            else:
                new_content = str(sys_msg.content) + addition
            sys_msg = SystemMessage(content=new_content)
        return request.override(system_message=sys_msg)

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        modified_req = self._modify_request(request)
        return handler(modified_req)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelResponse:
        modified_req = self._modify_request(request)
        return await handler(modified_req)
