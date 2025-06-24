import uuid
import time
import json
import gradio as gr
import modelscope_studio.components.antd as antd
import modelscope_studio.components.antdx as antdx
import modelscope_studio.components.base as ms
import modelscope_studio.components.pro as pro
from config import DEFAULT_LOCALE, DEFAULT_SETTINGS, DEFAULT_THEME, DEFAULT_SUGGESTIONS, save_history, get_text, user_config, bot_config, welcome_config, api_key, MODEL_OPTIONS_MAP

from ui_components.logo import Logo
from ui_components.settings_header import SettingsHeader
from ui_components.thinking_button import ThinkingButton


def format_history(history, sys_prompt):
    messages = []
    for item in history:
        if item["role"] == "user":
            # Use original_content if available, otherwise fall back to content
            content_to_process = item.get("original_content", item["content"])

            # Handle text and image content
            if isinstance(content_to_process, str):
                messages.append(
                    {"role": "user", "content": content_to_process})
            elif isinstance(content_to_process, dict):
                # Handle multimodal content (text + images)
                content_parts = []
                if content_to_process.get("text"):
                    content_parts.append(
                        {"type": "text", "text": content_to_process["text"]})
                if content_to_process.get("files"):
                    for file_info in content_to_process["files"]:
                        # Convert uploaded file to base64 for OpenAI API
                        import base64

                        # Handle different file path formats
                        file_path = None
                        if isinstance(file_info, dict):
                            file_path = file_info.get("path")
                        elif hasattr(file_info, 'path'):
                            file_path = file_info.path
                        elif hasattr(file_info, 'name'):
                            file_path = file_info.name
                        else:
                            file_path = str(file_info)

                        if not file_path:
                            print(
                                f"Warning: Could not get file path from {file_info}")
                            continue

                        try:
                            with open(file_path, "rb") as f:
                                image_data = base64.b64encode(
                                    f.read()).decode('utf-8')
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                            })
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")
                            continue
                messages.append({"role": "user", "content": content_parts})
        elif item["role"] == "assistant":
            contents = [{
                "type": "text",
                "text": content["content"]
            } for content in item["content"] if content["type"] == "text"]
            messages.append({
                "role": "assistant",
                "content": contents[0]["text"] if len(contents) > 0 else ""
            })
    return messages


def call_chat_bot(model, messages, enable_thinking, thinking_budget):
    """
    Centralized function to handle chat bot streaming API calls using OpenAI SDK
    """
    from openai import OpenAI

    # Initialize OpenAI client
    client = OpenAI(
        api_key="super-secret-token",
        base_url="https://styleme--example-vllm-openai-compatible-serve.modal.run/v1",
    )

    # Create streaming chat completion
    stream_response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=2048,
        temperature=0.9,
    )

    return stream_response


class Gradio_Events:

    @staticmethod
    def submit(state_value):

        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]
        settings = state_value["conversation_contexts"][
            state_value["conversation_id"]]["settings"]
        enable_thinking = state_value["conversation_contexts"][
            state_value["conversation_id"]]["enable_thinking"]
        model = settings.get("model")
        messages = format_history(
            history, sys_prompt=settings.get("sys_prompt", ""))

        history.append(
            {
                "role": "assistant",
                "content": [],
                "key": str(uuid.uuid4()),
                "header": MODEL_OPTIONS_MAP.get(model, {}).get("label", None),
                "loading": True,
                "status": "pending"
            }
        )

        yield {
            chatbot: gr.update(value=history),
            state: gr.update(value=state_value),
        }
        try:
            print("model: ", model, "-", "messages: ", messages)
            # Use the centralized call_chat_bot function for streaming
            response_stream = call_chat_bot(
                model=model,
                messages=messages,
                enable_thinking=enable_thinking,
                thinking_budget=settings.get("thinking_budget", 1) * 1024
            )

            start_time = time.time()
            reasoning_content = ""
            answer_content = ""
            is_thinking = False
            is_answering = False
            contents = [None, None]

            for chunk in response_stream:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content'):
                    if not is_thinking:
                        contents[0] = {
                            "type": "tool",
                            "content": "",
                            "options": {
                                "title": get_text("Thinking...", "Đang suy nghĩ..."),
                                "status": "pending"
                            },
                            "copyable": False,
                            "editable": False
                        }
                        is_thinking = True
                    reasoning_content += delta.get("reasoning_content", None)
                if hasattr(delta, 'content'):
                    if not is_answering:
                        thought_cost_time = "{:.2f}".format(
                            time.time() - start_time)
                        if contents[0]:
                            contents[0]["options"]["title"] = get_text(
                                f"End of Thought ({thought_cost_time}s)",
                                f"Đã suy nghĩ ({thought_cost_time}s)")
                            contents[0]["options"]["status"] = "done"
                        contents[1] = {
                            "type": "text",
                            "content": "",
                        }

                        is_answering = True
                    answer_content += delta.content

                if contents[0]:
                    contents[0]["content"] = reasoning_content
                if contents[1]:
                    contents[1]["content"] = answer_content

                # Update the history with the new content from assistant
                history[-1]["content"] = [
                    content for content in contents if content
                ]

                history[-1]["loading"] = False
                yield {
                    chatbot: gr.update(value=history),
                    state: gr.update(value=state_value)
                }

            print("model: ", model, "-", "reasoning_content: ",
                  reasoning_content, "\n", "content: ", answer_content)
            history[-1]["status"] = "done"
            cost_time = "{:.2f}".format(time.time() - start_time)
            history[-1]["footer"] = get_text(f"{cost_time}s",
                                             f"Thời gian: {cost_time}s")
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value),
            }
        except Exception as e:
            print("model: ", model, "-", "Error: ", e)
            history[-1]["loading"] = False
            history[-1]["status"] = "done"
            history[-1]["content"] += [{
                "type":
                "text",
                "content":
                f'<span style="color: var(--color-red-500)">{str(e)}</span>'
            }]
            yield {
                chatbot: gr.update(value=history),
                state: gr.update(value=state_value)
            }
            raise e

    @staticmethod
    def add_message(input_value, selected_images, settings_form_value, thinking_btn_state_value, state_value):
        if not state_value["conversation_id"]:
            random_id = str(uuid.uuid4())
            history = []
            state_value["conversation_id"] = random_id
            state_value["conversation_contexts"][
                state_value["conversation_id"]] = {
                    "history": history
            }
            # Use first part of message for conversation label
            label = input_value if isinstance(
                input_value, str) else "New Conversation"
            state_value["conversations"].append({
                "label": label[:50] + "..." if len(label) > 50 else label,
                "key": random_id
            })

        history = state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"]

        state_value["conversation_contexts"][
            state_value["conversation_id"]] = {
                "history": history,
                "settings": settings_form_value,
                "enable_thinking": thinking_btn_state_value["enable_thinking"]
        }

        # Use the selected_images from the state
        normalized_files = selected_images if selected_images else []

        # Create multimodal content if images are provided
        if normalized_files and len(normalized_files) > 0:
            import base64

            # Simple string-based approach (keeping the working method)
            display_content = input_value if input_value else ""

            for i, file in enumerate(normalized_files):
                try:
                    # Handle different file object structures
                    file_path = None
                    if hasattr(file, 'name'):
                        file_path = file.name
                    elif isinstance(file, dict):
                        file_path = file.get('path') or file.get('name')

                    if not file_path:
                        print(f"Warning: Could not get file path for file {i}")
                        continue

                    with open(file_path, "rb") as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')

                    # Determine proper MIME type
                    file_ext = file_path.lower().split('.')[-1]
                    mime_type = {
                        'jpg': 'image/jpeg',
                        'jpeg': 'image/jpeg',
                        'png': 'image/png',
                        'gif': 'image/gif',
                        'webp': 'image/webp'
                    }.get(file_ext, 'image/jpeg')

                    data_url = f"data:{mime_type};base64,{image_data}"

                    # Add image as HTML img tag with professional styling
                    display_content += f'\n\n<img src="{data_url}" style="max-width: 300px; max-height: 300px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 8px 0;" alt="Uploaded image {i+1}"/>'
                except Exception as e:
                    print(f"Error processing image {file}: {e}")
                    file_name = str(file).split(
                        '/')[-1] if isinstance(file, str) else "unknown file"
                    display_content += f"\n❌ Error loading image: {file_name}"

            message_content = display_content

            # Store the original format for API calls
            original_content = {
                "text": input_value,
                "files": [{"path": (f.name if hasattr(f, 'name') else f.get('path', f.get('name', str(f))))} for f in normalized_files]
            }
        else:
            # Text-only message
            message_content = input_value
            original_content = input_value

        # Handle multimodal input (text + images)
        history.append({
            "role": "user",
            "content": message_content,
            "original_content": original_content,  # Store original for API calls
            "key": str(uuid.uuid4())
        })

        yield Gradio_Events.preprocess_submit(clear_input=True)(state_value)

        try:
            for chunk in Gradio_Events.submit(state_value):
                yield chunk
        except Exception as e:
            raise e
        finally:
            yield Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def preprocess_submit(clear_input=True):
        def preprocess_submit_handler(state_value):
            history = state_value["conversation_contexts"][
                state_value["conversation_id"]]["history"]
            return {
                **(
                    {
                        input: gr.update(value=None, loading=True) if clear_input else gr.update(loading=True),
                        # Keep image upload hidden and clear it
                        image_upload: gr.update(value=None, visible=False),
                    } if clear_input else {}
                ),
                conversations: gr.update(
                    active_key=state_value["conversation_id"],
                    items=list(
                        map(
                            lambda item: {
                                **item,
                                "disabled": True if item["key"] != state_value[
                                    "conversation_id"] else False,
                            },
                            state_value["conversations"]
                        )
                    )
                ),
                add_conversation_btn: gr.update(disabled=True),
                clear_btn: gr.update(disabled=True),
                conversation_delete_menu_item: gr.update(disabled=True),
                # Disable upload button during processing
                upload_btn: gr.update(disabled=True),
                chatbot: gr.update(
                    value=history,
                    bot_config=bot_config(
                        disabled_actions=['edit', 'retry', 'delete']),
                    user_config=user_config(
                        disabled_actions=['edit', 'delete'])
                ),
                state: gr.update(value=state_value),
            }
        return preprocess_submit_handler

    @staticmethod
    def postprocess_submit(state_value):
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        return {
            input: gr.update(loading=False),
            # Keep image upload hidden and cleared
            image_upload: gr.update(value=None, visible=False),
            conversation_delete_menu_item: gr.update(disabled=False),
            clear_btn: gr.update(disabled=False),
            conversations: gr.update(items=state_value["conversations"]),
            add_conversation_btn: gr.update(disabled=False),
            upload_btn: gr.update(disabled=False),  # Re-enable upload button
            selected_images_state: gr.update(value=[]),
            chatbot: gr.update(
                value=history,
                bot_config=bot_config(),
                user_config=user_config()
            ),
            state: gr.update(value=state_value),
        }

    @staticmethod
    def cancel(state_value):
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history[-1]["loading"] = False
        history[-1]["status"] = "done"
        history[-1]["footer"] = get_text("Chat completion paused",
                                         "Cuộc trò chuyện đã bị tạm dừng")
        return {
            **Gradio_Events.postprocess_submit(state_value),
            # Keep upload area hidden
            image_upload: gr.update(value=None, visible=False),
            selected_images_state: gr.update(value=[])  # Clear selected images
        }

    @staticmethod
    def delete_message(state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history = history[:index] + history[index + 1:]
        state_value["conversation_contexts"][state_value["conversation_id"]
                                             ]["history"] = history

        return gr.update(value=state_value)

    @staticmethod
    def edit_message(state_value, chatbot_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history[index]["content"] = chatbot_value[index]["content"]
        return gr.update(value=state_value)

    @staticmethod
    def regenerate_message(settings_form_value, thinking_btn_state_value,
                           state_value, e: gr.EventData):
        index = e._data["payload"][0]["index"]
        history = state_value["conversation_contexts"][state_value["conversation_id"]]["history"]
        history = history[:index]

        state_value["conversation_contexts"][
            state_value["conversation_id"]] = {
                "history": history,
                "settings": settings_form_value,
                "enable_thinking": thinking_btn_state_value["enable_thinking"]
        }

        yield Gradio_Events.preprocess_submit()(state_value)
        try:
            for chunk in Gradio_Events.submit(state_value):
                yield chunk
        except Exception as e:
            raise e
        finally:
            yield Gradio_Events.postprocess_submit(state_value)

    @staticmethod
    def select_suggestion(input_value, e: gr.EventData):
        input_value = input_value[:-1] + e._data["payload"][0]
        return gr.update(value=input_value)

    @staticmethod
    def apply_prompt(e: gr.EventData):
        return gr.update(value=e._data["payload"][0]["value"]["description"])

    @staticmethod
    def new_chat(thinking_btn_state, state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversation_id"] = ""
        thinking_btn_state["enable_thinking"] = True
        return gr.update(active_key=state_value["conversation_id"]), gr.update(
            value=None), gr.update(value=DEFAULT_SETTINGS), gr.update(
                value=thinking_btn_state), gr.update(value=state_value)

    @staticmethod
    def select_conversation(thinking_btn_state_value, state_value,
                            e: gr.EventData):
        active_key = e._data["payload"][0]
        if state_value["conversation_id"] == active_key or (
                active_key not in state_value["conversation_contexts"]):
            return gr.skip()
        state_value["conversation_id"] = active_key
        thinking_btn_state_value["enable_thinking"] = state_value[
            "conversation_contexts"][active_key]["enable_thinking"]
        return gr.update(active_key=active_key), gr.update(
            value=state_value["conversation_contexts"][active_key]["history"]
        ), gr.update(value=state_value["conversation_contexts"][active_key]
                     ["settings"]), gr.update(
                         value=thinking_btn_state_value), gr.update(
                             value=state_value)

    @staticmethod
    def click_conversation_menu(state_value, e: gr.EventData):
        conversation_id = e._data["payload"][0]["key"]
        operation = e._data["payload"][1]["key"]
        if operation == "delete":
            del state_value["conversation_contexts"][conversation_id]

            state_value["conversations"] = [
                item for item in state_value["conversations"]
                if item["key"] != conversation_id
            ]

            if state_value["conversation_id"] == conversation_id:
                state_value["conversation_id"] = ""
                return gr.update(
                    items=state_value["conversations"],
                    active_key=state_value["conversation_id"]), gr.update(
                        value=None), gr.update(value=state_value)
            else:
                return gr.update(
                    items=state_value["conversations"]), gr.skip(), gr.update(
                        value=state_value)
        return gr.skip()

    @staticmethod
    def toggle_settings_header(settings_header_state_value):
        settings_header_state_value[
            "open"] = not settings_header_state_value["open"]
        return gr.update(value=settings_header_state_value)

    @staticmethod
    def clear_conversation_history(state_value):
        if not state_value["conversation_id"]:
            return gr.skip()
        state_value["conversation_contexts"][
            state_value["conversation_id"]]["history"] = []
        return gr.update(value=None), gr.update(value=state_value)

    @staticmethod
    def update_browser_state(state_value):

        return gr.update(value=dict(
            conversations=state_value["conversations"],
            conversation_contexts=state_value["conversation_contexts"]))

    @staticmethod
    def apply_browser_state(browser_state_value, state_value):
        state_value["conversations"] = browser_state_value["conversations"]
        state_value["conversation_contexts"] = browser_state_value[
            "conversation_contexts"]
        return gr.update(
            items=browser_state_value["conversations"]), gr.update(
                value=state_value)

    @staticmethod
    def trigger_file_upload():
        """Dummy function - actual file browser trigger is handled by JavaScript"""
        return gr.update()


css = """
.gradio-container {
  padding: 0 !important;
}
.gradio-container > main.fillable {
  padding: 0 !important;
}
#chatbot {
  height: calc(100vh - 21px - 16px);
  max-height: 1500px;
}
#chatbot .chatbot-conversations {
  height: 100vh;
  background-color: var(--ms-gr-ant-color-bg-layout);
  padding-left: 4px;
  padding-right: 4px;
}
#chatbot .chatbot-conversations .chatbot-conversations-list {
  padding-left: 0;
  padding-right: 0;
}
#chatbot .chatbot-chat {
  padding: 32px;
  padding-bottom: 0;
  height: 100%;
}
@media (max-width: 768px) {
  #chatbot .chatbot-chat {
      padding: 0;
  }
}
#chatbot .chatbot-chat .chatbot-chat-messages {
  flex: 1;
}
#chatbot .setting-form-thinking-budget .ms-gr-ant-form-item-control-input-content {
    display: flex;
    flex-wrap: wrap;
}

/* Hidden image upload component */
#hidden-image-upload {
    display: none !important;
}

/* Image preview area styling */
#image-preview-area {
    min-height: 0px !important;
    border-radius: 8px;
    transition: all 0.3s ease;
    flex-wrap: wrap !important;
    gap: 8px !important;
}

#image-preview-area:not(:empty) {
    background-color: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    padding: 8px 12px !important;
    margin-bottom: 8px !important;
}

/* Image display in chat */
.chatbot-chat-messages img {
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: transform 0.2s ease;
}

.chatbot-chat-messages img:hover {
    transform: scale(1.02);
}

/* Upload button styling */
#upload-trigger-btn {
    transition: all 0.2s ease;
}

#upload-trigger-btn:hover {
    background-color: rgba(22, 119, 255, 0.06) !important;
    color: #1677ff !important;
}
"""

model_options_map_json = json.dumps(MODEL_OPTIONS_MAP)
js = """
function init() { 
    window.MODEL_OPTIONS_MAP = """ + model_options_map_json + """;
    
    // Store selected files globally
    window.selectedImageFiles = [];
    window.fileInputInitialized = false;
    
    // Function to create image preview thumbnails
    window.createImagePreview = function(files) {
        const previewArea = document.querySelector('#image-preview-area');
        if (!previewArea) {
            console.log('Preview area not found');
            return;
        }
        
        console.log('Creating preview for', files, 'files');
        
        // Clear previous previews
        previewArea.innerHTML = '';
        
        // Handle different file input formats
        let fileArray = [];
        if (!files) {
            // No files
            previewArea.style.display = 'none';
            return;
        } else if (Array.isArray(files)) {
            fileArray = files;
        } else if (files.length !== undefined) {
            // FileList or similar array-like object
            fileArray = Array.from(files);
        } else if (typeof files === 'object') {
            // Single file object
            fileArray = [files];
        } else {
            console.log('Unexpected file format:', files);
            previewArea.style.display = 'none';
            return;
        }
        
        window.selectedImageFiles = fileArray;
        
        if (fileArray.length === 0) {
            previewArea.style.display = 'none';
            return;
        }
        
        console.log('Processing', fileArray.length, 'files');
        
        // Show preview area with proper styling
        previewArea.style.display = 'flex';
        previewArea.style.backgroundColor = '#f8f9fa';
        previewArea.style.border = '1px solid #e9ecef';
        previewArea.style.borderRadius = '8px';
        previewArea.style.padding = '8px 12px';
        previewArea.style.marginBottom = '8px';
        previewArea.style.flexWrap = 'wrap';
        previewArea.style.gap = '8px';
        
        fileArray.forEach((file, index) => {
            // Handle different file object formats
            let fileName = 'unknown';
            let filePath = null;
            
            if (file instanceof File) {
                // Regular File object
                fileName = file.name;
                const reader = new FileReader();
                reader.onload = function(e) {
                    window.createImageThumbnail(e.target.result, fileName, index, previewArea);
                };
                reader.readAsDataURL(file);
            } else if (file && typeof file === 'object') {
                // Gradio file object or similar
                fileName = file.name || file.path || 'unknown';
                if (fileName.includes('/')) {
                    fileName = fileName.split('/').pop();
                }
                filePath = file.name || file.path;
                
                if (filePath) {
                    // Try to use the file path directly as image URL
                    window.createImageThumbnail(filePath, fileName, index, previewArea);
                } else {
                    // Fallback: show file name only
                    window.createImageThumbnail('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA4MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjgwIiBoZWlnaHQ9IjgwIiBmaWxsPSIjRjVGNUY1Ii8+CjxwYXRoIGQ9Ik00MCA0MEwyNSAyNUw1NSA1NUw0MCA0MFoiIGZpbGw9IiNEOUQ5RDkiLz4KPHN2Zz4K', fileName, index, previewArea);
                }
            } else {
                console.log('Unexpected file object:', file);
            }
        });
    };
    
    // Helper function to create image thumbnail
    window.createImageThumbnail = function(imageSrc, fileName, index, previewArea) {
        const imageContainer = document.createElement('div');
        imageContainer.style.cssText = `
            position: relative;
            display: inline-block;
            margin: 2px;
        `;
        
        const img = document.createElement('img');
        img.src = imageSrc;
        img.style.cssText = `
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 6px;
            border: 1px solid #d9d9d9;
            cursor: pointer;
            transition: all 0.2s ease;
        `;
        
        // Add hover effect
        img.onmouseover = function() {
            this.style.transform = 'scale(1.05)';
            this.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
        };
        img.onmouseout = function() {
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'none';
        };
        
        const removeBtn = document.createElement('div');
        removeBtn.innerHTML = '×';
        removeBtn.style.cssText = `
            position: absolute;
            top: -6px;
            right: -6px;
            width: 20px;
            height: 20px;
            background: #ff4d4f;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            transition: all 0.2s ease;
        `;
        
        // Add hover effect to remove button
        removeBtn.onmouseover = function() {
            this.style.backgroundColor = '#ff7875';
            this.style.transform = 'scale(1.1)';
        };
        removeBtn.onmouseout = function() {
            this.style.backgroundColor = '#ff4d4f';
            this.style.transform = 'scale(1)';
        };
        
        removeBtn.onclick = function(e) {
            e.stopPropagation();
            window.removeImageFromPreview(index);
        };
        
        // Add file name tooltip
        const displayName = fileName.length > 20 ? fileName.substring(0, 20) + '...' : fileName;
        img.title = displayName;
        
        imageContainer.appendChild(img);
        imageContainer.appendChild(removeBtn);
        previewArea.appendChild(imageContainer);
    };
    
    // Function to remove image from preview
    window.removeImageFromPreview = function(index) {
        window.selectedImageFiles.splice(index, 1);
        
        // Update preview without triggering change events
        window.createImagePreview(window.selectedImageFiles);
        
        // Update the hidden file input value only (without triggering events)
        const hiddenUpload = document.querySelector('#hidden-image-upload input[type="file"]');
        if (hiddenUpload) {
            // Create a new FileList-like object with remaining files
            const dt = new DataTransfer();
            window.selectedImageFiles.forEach(file => {
                dt.items.add(file);
            });
            hiddenUpload.files = dt.files;
        }
    };
    
    // Function to clear all image previews
    window.clearImagePreviews = function() {
        console.log('Clearing image previews');
        window.selectedImageFiles = [];
        const previewArea = document.querySelector('#image-preview-area');
        if (previewArea) {
            previewArea.innerHTML = '';
            previewArea.style.display = 'none';
        }
        
        // Clear the hidden file input without triggering events
        const hiddenUpload = document.querySelector('#hidden-image-upload input[type="file"]');
        if (hiddenUpload) {
            hiddenUpload.value = '';
        }
    };
    
    // Function to trigger file upload directly
    window.triggerFileUpload = function() {
        console.log('Triggering file upload');
        const hiddenUpload = document.querySelector('#hidden-image-upload input[type="file"]');
        if (hiddenUpload) {
            hiddenUpload.click();
        } else {
            console.log('Hidden upload input not found');
        }
    };
    
    // Function to find and attach listener to file input (only once)
    window.attachFileListener = function() {
        if (window.fileInputInitialized) {
            return true;
        }
        
        // Try multiple selectors to find the file input
        const selectors = [
            '#hidden-image-upload input[type="file"]',
            '#hidden-image-upload input',
            'input[type="file"][data-testid]',
            'input[type="file"]'
        ];
        
        let fileInput = null;
        for (let selector of selectors) {
            const elements = document.querySelectorAll(selector);
            for (let element of elements) {
                // Check if this is likely our hidden upload input
                if (element.closest('#hidden-image-upload') || 
                    element.accept && element.accept.includes('image') ||
                    element.multiple) {
                    fileInput = element;
                    break;
                }
            }
            if (fileInput) break;
        }
        
        if (fileInput) {
            console.log('Attaching listener to file input:', fileInput);
            
            // Remove any existing listeners first
            fileInput.removeEventListener('change', window.fileChangeHandler);
            
            // Define the handler function
            window.fileChangeHandler = function(e) {
                console.log('File input changed:', e.target.files);
                if (e.target.files && e.target.files.length > 0) {
                    window.selectedImageFiles = Array.from(e.target.files);
                    window.createImagePreview(e.target.files);
                } else {
                    window.clearImagePreviews();
                }
            };
            
            // Add the event listener
            fileInput.addEventListener('change', window.fileChangeHandler, { passive: true });
            
            window.fileInputInitialized = true;
            return true;
        }
        return false;
    };
    
    // Function to setup upload button click handler (only once)
    window.setupUploadButton = function() {
        const uploadBtn = document.querySelector('#upload-trigger-btn');
        if (uploadBtn && !uploadBtn.hasAttribute('data-click-attached')) {
            console.log('Setting up upload button');
            uploadBtn.setAttribute('data-click-attached', 'true');
            uploadBtn.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                window.triggerFileUpload();
            }, { once: false, passive: false });
        }
    };
    
    // Main initialization function
    window.initImageUpload = function() {
        console.log('Initializing image upload functionality');
        
        // Setup upload button
        window.setupUploadButton();
        
        // Try to attach file listener
        if (!window.attachFileListener()) {
            // If not found, try once more after a short delay
            setTimeout(function() {
                if (!window.fileInputInitialized) {
                    window.attachFileListener();
                }
            }, 1000);
        }
    };
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.initImageUpload);
    } else {
        window.initImageUpload();
    }
    
    // Multiple retries to catch dynamically created elements
    let retryCount = 0;
    const maxRetries = 5;
    const retryInterval = 1000;
    
    function retryInit() {
        if (!window.fileInputInitialized && retryCount < maxRetries) {
            console.log(`Retry ${retryCount + 1}/${maxRetries}: Initializing image upload`);
            window.initImageUpload();
            retryCount++;
            setTimeout(retryInit, retryInterval);
        } else if (window.fileInputInitialized) {
            console.log('Image upload initialization successful');
        } else {
            console.log('Image upload initialization failed after all retries');
        }
    }
    
    // Start retry process
    setTimeout(retryInit, retryInterval);
}
"""

with gr.Blocks(css=css, js=js, fill_width=True) as demo:
    state = gr.State({
        "conversation_contexts": {},
        "conversations": [],
        "conversation_id": "",
    })

    # Hidden state to store selected images information
    selected_images_state = gr.State([])

    with ms.Application(), antdx.XProvider(theme=DEFAULT_THEME, locale=DEFAULT_LOCALE), ms.AutoLoading():
        with antd.Row(gutter=[20, 20], wrap=False, elem_id="chatbot"):
            # Left Column
            with antd.Col(md=dict(flex="0 0 260px", span=24, order=0),
                          span=0,
                          elem_style=dict(width=0),
                          order=1):
                with ms.Div(elem_classes="chatbot-conversations"):
                    with antd.Flex(vertical=True,
                                   gap="small",
                                   elem_style=dict(height="100%")):
                        # Logo
                        Logo()

                        # New Conversation Button
                        with antd.Button(value=None,
                                         color="primary",
                                         variant="filled",
                                         block=True) as add_conversation_btn:
                            ms.Text(get_text("New Conversation",
                                    "Cuộc trò chuyện mới"))
                            with ms.Slot("icon"):
                                antd.Icon("PlusOutlined")

                        # Conversations List
                        with antdx.Conversations(
                                elem_classes="chatbot-conversations-list",
                        ) as conversations:
                            with ms.Slot('menu.items'):
                                with antd.Menu.Item(
                                        label="Delete", key="delete",
                                        danger=True
                                ) as conversation_delete_menu_item:
                                    with ms.Slot("icon"):
                                        antd.Icon("DeleteOutlined")

            # Right Column
            with antd.Col(flex=1, elem_style=dict(height="100%")):
                with antd.Flex(vertical=True,
                               gap="small",
                               elem_classes="chatbot-chat"):
                    # Chatbot
                    chatbot = pro.Chatbot(elem_classes="chatbot-chat-messages",
                                          height=0,
                                          welcome_config=welcome_config(),
                                          user_config=user_config(),
                                          bot_config=bot_config())

                    # Input section with separate image upload
                    with antd.Flex(vertical=True, gap="small"):
                        # Image upload area (hidden - triggered by button)
                        image_upload = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="Upload Images",
                            visible=False,
                            elem_id="hidden-image-upload"
                        )

                        # Image preview area (shows uploaded images like ChatGPT)
                        with antd.Flex(
                            gap="small",
                            wrap=True,
                            elem_id="image-preview-area",
                            elem_style=dict(
                                minHeight="0px",
                                padding="8px 12px",
                                borderRadius="8px",
                                backgroundColor="transparent",
                                display="none"
                            )
                        ) as image_preview_container:
                            pass  # This will be populated by JavaScript

                        # Text input with upload button
                        with antdx.Suggestion(
                                items=DEFAULT_SUGGESTIONS,
                                should_trigger="""(e, { onTrigger, onKeyDown }) => {
                          switch(e.key) {
                            case '/':
                              onTrigger()
                              break
                            case 'ArrowRight':
                            case 'ArrowLeft':
                            case 'ArrowUp':
                            case 'ArrowDown':
                              break;
                            default:
                              onTrigger(false)
                          }
                          onKeyDown(e)
                        }""") as suggestion:
                            with ms.Slot("children"):
                                with antdx.Sender(
                                    placeholder=get_text(
                                        "Enter your message here...",
                                        "Nhập tin nhắn của bạn tại đây..."
                                    )
                                ) as input:
                                    with ms.Slot("header"):
                                        settings_header_state, settings_form = SettingsHeader()
                                    with ms.Slot("prefix"):
                                        with antd.Flex(
                                                gap=4,
                                                wrap=True,
                                                elem_style=dict(maxWidth='40vw')):
                                            with antd.Button(
                                                    value=None,
                                                    type="text") as setting_btn:
                                                with ms.Slot("icon"):
                                                    antd.Icon(
                                                        "SettingOutlined")
                                            with antd.Button(
                                                    value=None,
                                                    type="text") as clear_btn:
                                                with ms.Slot("icon"):
                                                    antd.Icon("ClearOutlined")

                                            # Image upload button - triggers file browser directly
                                            with antd.Button(
                                                    value=None,
                                                    type="text",
                                                    elem_id="upload-trigger-btn") as upload_btn:
                                                with ms.Slot("icon"):
                                                    antd.Icon(
                                                        "PictureOutlined")

                                            thinking_btn_state = ThinkingButton()

    # Events Handler
    # Browser State Handler
    if save_history:
        browser_state = gr.BrowserState(
            {
                "conversation_contexts": {},
                "conversations": [],
            },
            storage_key="qwen2.5_chat_demo_storage")
        state.change(fn=Gradio_Events.update_browser_state,
                     inputs=[state],
                     outputs=[browser_state])

        demo.load(fn=Gradio_Events.apply_browser_state,
                  inputs=[browser_state, state],
                  outputs=[conversations, state])

    # Conversations Handler
    add_conversation_btn.click(fn=Gradio_Events.new_chat,
                               inputs=[thinking_btn_state, state],
                               outputs=[
                                   conversations, chatbot, settings_form,
                                   thinking_btn_state, state
                               ])
    conversations.active_change(fn=Gradio_Events.select_conversation,
                                inputs=[thinking_btn_state, state],
                                outputs=[
                                    conversations, chatbot, settings_form,
                                    thinking_btn_state, state
                                ])
    conversations.menu_click(fn=Gradio_Events.click_conversation_menu,
                             inputs=[state],
                             outputs=[conversations, chatbot, state])
    # Chatbot Handler
    chatbot.welcome_prompt_select(fn=Gradio_Events.apply_prompt,
                                  outputs=[input])

    chatbot.delete(fn=Gradio_Events.delete_message,
                   inputs=[state],
                   outputs=[state])
    chatbot.edit(fn=Gradio_Events.edit_message,
                 inputs=[state, chatbot],
                 outputs=[state])

    regenerating_event = chatbot.retry(
        fn=Gradio_Events.regenerate_message,
        inputs=[settings_form, thinking_btn_state, state],
        outputs=[
            input, image_upload, clear_btn, conversation_delete_menu_item,
            add_conversation_btn, conversations, chatbot, upload_btn, selected_images_state, state
        ])

    # Input Handler
    submit_event = input.submit(
        fn=Gradio_Events.add_message,
        inputs=[input, selected_images_state,
                settings_form, thinking_btn_state, state],
        outputs=[
            input, image_upload, clear_btn, conversation_delete_menu_item,
            add_conversation_btn, conversations, chatbot, upload_btn, selected_images_state, state
        ])
    input.cancel(fn=Gradio_Events.cancel,
                 inputs=[state],
                 outputs=[
                     input, image_upload, conversation_delete_menu_item, clear_btn,
                     conversations, add_conversation_btn, chatbot, upload_btn, selected_images_state, state
                 ],
                 cancels=[submit_event, regenerating_event],
                 queue=False)
    # Input Actions Handler
    setting_btn.click(fn=Gradio_Events.toggle_settings_header,
                      inputs=[settings_header_state],
                      outputs=[settings_header_state])
    clear_btn.click(fn=Gradio_Events.clear_conversation_history,
                    inputs=[state],
                    outputs=[chatbot, state])
    suggestion.select(fn=Gradio_Events.select_suggestion,
                      inputs=[input],
                      outputs=[input])

    # Upload button click handler - JavaScript handles the actual file browser trigger
    upload_btn.click(
        fn=Gradio_Events.trigger_file_upload,
        outputs=[],
        js="window.triggerFileUpload()"
    )

    # Handle file upload changes to show preview only
    image_upload.change(
        fn=lambda files: files if files else [],
        inputs=[image_upload],
        outputs=[selected_images_state],
        queue=False
    )

    # Clear image previews and state when input is submitted
    input.submit(
        fn=lambda: [],
        outputs=[selected_images_state],
        js="setTimeout(() => window.clearImagePreviews(), 100)",
        queue=False
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=100,
               max_size=100).launch(share=True, ssr_mode=False, max_threads=100)
