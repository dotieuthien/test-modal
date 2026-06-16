import ast
import re

class LLMOutputParser:
    @staticmethod
    def literal_eval(response_content: str):
        response_content = response_content.strip()

        # remove content between <think> and </think>, especial for DeepSeek reasoning model
        if "<think>" in response_content and "</think>" in response_content:
            end_of_think = response_content.find("</think>") + len("</think>")
            response_content = response_content[end_of_think:]

        try:
            if response_content.startswith("```") and response_content.endswith("```"):
                if response_content.startswith("```python"):
                    response_content = response_content[9:-3]
                elif response_content.startswith("```json"):
                    response_content = response_content[7:-3]
                elif response_content.startswith("```str"):
                    response_content = response_content[6:-3]
                elif response_content.startswith("```\n"):
                    response_content = response_content[4:-3]
                else:
                    raise ValueError("Invalid code block format")
            result = ast.literal_eval(response_content.strip())
        except Exception:
            # Try to find JSON/list pattern
            matches = re.findall(r"(\[.*?\]|\{.*?\})", response_content, re.DOTALL)

            if len(matches) == 1:
                json_part = matches[0]
                return ast.literal_eval(json_part)

            # Try to parse markdown bullet list (-, *, •)
            lines = response_content.strip().split('\n')
            bullet_items = []
            for line in lines:
                line = line.strip()
                # Match lines starting with *, -, •, or numbered list
                if re.match(r'^[\*\-\•]\s+', line):
                    # Remove bullet point
                    item = re.sub(r'^[\*\-\•]\s+', '', line).strip()
                    bullet_items.append(item)
                elif re.match(r'^\d+[\.\)]\s+', line):
                    # Remove numbered list marker
                    item = re.sub(r'^\d+[\.\)]\s+', '', line).strip()
                    bullet_items.append(item)

            if bullet_items:
                return bullet_items

            raise ValueError(
                f"Invalid JSON/List format for response content:\n{response_content}"
            )

        return result
