import os
from modelscope_studio.components.pro.chatbot import ChatbotActionConfig, ChatbotBotConfig, ChatbotUserConfig, ChatbotWelcomeConfig

# Env
is_vn = False
api_key = os.getenv('API_KEY')


def get_text(text: str, vi_text: str):
    if is_vn:
        return vi_text
    return text


# Save history in browser
save_history = True


# Chatbot Config
def user_config(disabled_actions=None):
    return ChatbotUserConfig(
        class_names=dict(content="user-message-content"),
        actions=[
            "copy", "edit",
            ChatbotActionConfig(
                action="delete",
                popconfirm=dict(title=get_text("Delete the message", "Xóa tin nhắn"),
                                description=get_text(
                                    "Are you sure to delete this message?",
                                    "Bạn có chắc chắn muốn xóa tin nhắn này không?"),
                                okButtonProps=dict(danger=True)))
        ],
        disabled_actions=disabled_actions)


def bot_config(disabled_actions=None):
    return ChatbotBotConfig(actions=[
        "copy", "edit",
        ChatbotActionConfig(
            action="retry",
            popconfirm=dict(
                title=get_text("Regenerate the message", "Tạo lại tin nhắn"),
                description=get_text(
                    "Regenerate the message will also delete all subsequent messages.",
                    "Tạo lại tin nhắn sẽ xóa tất cả các tin nhắn tiếp theo."),
                okButtonProps=dict(danger=True))),
        ChatbotActionConfig(action="delete",
                            popconfirm=dict(
                                title=get_text("Delete the message", "Xóa tin nhắn"),
                                description=get_text(
                                    "Are you sure to delete this message?",
                                    "Bạn có chắc chắn muốn xóa tin nhắn này không?"),
                                okButtonProps=dict(danger=True)))
    ],
                            avatar="gradio_ui/assets/qwen.png",
                            disabled_actions=disabled_actions)


def welcome_config():
    return ChatbotWelcomeConfig(
        variant="borderless",
        icon="gradio_ui/assets/qwen.png",
        title=get_text("Hello, I'm Qwen2.5", "Xin chào, tôi là Qwen2.5"),   
        description=get_text("Select a model and enter text to get started.",
                             "Chọn mô hình và nhập văn bản để bắt đầu cuộc trò chuyện."),
        prompts=dict(
            title=get_text("How can I help you today?", "Có thể giúp bạn được gì?"),
            styles={
                "list": {
                    "width": '100%',
                },
                "item": {
                    "flex": 1,
                },
            },
            items=[{
                "label":
                get_text("📅 Make a plan", "📅 制定计划"),
                "children": [{
                    "description":
                    get_text("Help me with a plan to start a business",
                             "Giúp tôi lập kế hoạch khởi nghiệp")
                }, {
                    "description":
                    get_text("Help me with a plan to achieve my goals",
                             "Giúp tôi lập kế hoạch đạt được mục tiêu")
                }, {
                    "description":
                    get_text("Help me with a plan for a successful interview",
                             "Giúp tôi lập kế hoạch cho một cuộc phỏng vấn thành công")
                }]
            }, {
                "label":
                get_text("🖋 Help me write", "🖋 帮我写"),
                "children": [{
                    "description":
                    get_text("Help me write a story with a twist ending",
                             "Giúp tôi viết một câu chuyện với kết thúc bất ngờ")
                }, {
                    "description":
                    get_text("Help me write a blog post on mental health",
                             "Giúp tôi viết một bài viết blog về sức khỏe tinh thần")
                }, {
                    "description":
                    get_text("Help me write a letter to my future self",
                             "Giúp tôi viết một bức thư cho tương lai")
                }]
            }]),
    )


DEFAULT_SUGGESTIONS = [{
    "label":
    get_text('Make a plan', 'Lập kế hoạch'),
    "value":
    get_text('Make a plan', 'Lập kế hoạch'),
    "children": [{
        "label":
        get_text("Start a business", "Khởi nghiệp"),
        "value":
        get_text("Help me with a plan to start a business", "Giúp tôi lập kế hoạch khởi nghiệp")
    }, {
        "label":
        get_text("Achieve my goals", "Đạt được mục tiêu"),
        "value":
        get_text("Help me with a plan to achieve my goals", "Giúp tôi lập kế hoạch đạt được mục tiêu")
    }, {
        "label":
        get_text("Successful interview", "Cuộc phỏng vấn thành công"),
        "value":
        get_text("Help me with a plan for a successful interview",
                 "Giúp tôi lập kế hoạch cho một cuộc phỏng vấn thành công")
    }]
}, {
    "label":
    get_text('Help me write', 'Giúp tôi viết'),
    "value":
    get_text("Help me write", 'Giúp tôi viết'),
    "children": [{
        "label":
        get_text("Story with a twist ending", "Câu chuyện với kết thúc bất ngờ"),
        "value":
        get_text("Help me write a story with a twist ending",
                 "Giúp tôi viết một câu chuyện với kết thúc bất ngờ")
    }, {
        "label":
        get_text("Blog post on mental health", "Bài viết blog về sức khỏe tinh thần"),
        "value":
        get_text("Help me write a blog post on mental health",
                 "Giúp tôi viết một bài viết blog về sức khỏe tinh thần")
    }, {
        "label":
        get_text("Letter to my future self", "Bức thư cho tương lai"),
        "value":
        get_text("Help me write a letter to my future self", "Giúp tôi viết một bức thư cho tương lai")
    }]
}]

DEFAULT_SYS_PROMPT = "You are a helpful and harmless assistant."

MIN_THINKING_BUDGET = 1

MAX_THINKING_BUDGET = 38

DEFAULT_THINKING_BUDGET = 38

DEFAULT_MODEL = "Qwen2.5-VL-7B-Instruct-AWQ"

MODEL_OPTIONS = [
    {
        "label": get_text("Qwen2.5-VL-7B-Instruct-AWQ", "Qwen2.5-VL-7B-Instruct-AWQ"),
        "modelId": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "value": "Qwen2.5-VL-7B-Instruct-AWQ"
    },
]

for model in MODEL_OPTIONS:
    model["link"] = f"https://huggingface.co/{model['modelId']}"

MODEL_OPTIONS_MAP = {model["value"]: model for model in MODEL_OPTIONS}

DEFAULT_LOCALE = 'vi_VN' if is_vn else 'en_US'

DEFAULT_THEME = {
    "token": {
        "colorPrimary": "#6A57FF",
    }
}

DEFAULT_SETTINGS = {
    "model": DEFAULT_MODEL,
    "sys_prompt": DEFAULT_SYS_PROMPT,
    "thinking_budget": DEFAULT_THINKING_BUDGET
}
