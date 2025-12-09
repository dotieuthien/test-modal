from typing import TypedDict, List, Union, Any

class Message(TypedDict):
    role: str
    content: Union[str, List[Any]]
