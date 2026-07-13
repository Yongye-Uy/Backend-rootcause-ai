REACT_TOOL_FORMAT_REMINDER = (
    "IMPORTANT: when you decide to use the search tool, you MUST respond using EXACTLY this "
    "plain-text format, not JSON:\n"
    "Thought: <your reasoning>\n"
    "Action: tavily_search\n"
    "Action Input: <your search query as a plain string, not an object>\n\n"
    "Do not wrap your action in a JSON object. Do not use curly braces for the action block. "
    "Some models default to a JSON tool-call format, but this system only recognizes the "
    "plain-text format above -- responding in JSON means the tool is never actually called."
)
