from openai import OpenAI
import openai
import os
import tempfile
import yaml
from PIL import Image
import requests
import base64
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
# ------------- CONFIG --------------

def load_config():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if os.path.exists('keyfile.yaml'):
        with open('keyfile.yaml', 'r') as f:
            keyfile = yaml.safe_load(f)
        config.update(keyfile) 
    return config

CONFIG = load_config()

client = OpenAI(api_key=CONFIG['open_ai_api_key'], base_url="https://datalab-api.reyrey.net/Api/OpenAI")

# ----------- TEXT CAPABILITIES -----------

def text_generation_demo():
    prompt = "Write a short, friendly email inviting a colleague to lunch."
    response = client.chat.completions.create(model=CONFIG['chat_model'],
    messages=[{"role": "user", "content": prompt}])
    print("\nPrompt:", prompt)
    print("Model output:", response.choices[0].message.content.strip(), "\n")

def conversational_ai_demo():
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
    response = client.chat.completions.create(model=CONFIG['chat_model'], messages=conversation)
    print("\nModel output:", response.choices[0].message.content.strip(), "\n")

def content_summarization_demo():
    long_text = (
        "OpenAI has developed advanced generative models for text, code, images, and more. "
        "These models are used in a variety of domains, including business, education, creative arts, and science..."
    )
    prompt = f"Please summarize the following: {long_text}"
    response = client.chat.completions.create(model=CONFIG['chat_model'],
    messages=[{"role": "user", "content": prompt}])
    print("\nSummary:", response.choices[0].message.content.strip(), "\n")

def code_generation_demo():
    prompt = "Write a Python function to compute the nth Fibonacci number."
    response = client.chat.completions.create(model=CONFIG['chat_model'],
    messages=[{"role": "user", "content": prompt}])
    print("\nCode:\n", response.choices[0].message.content.strip(), "\n")

def language_translation_demo():
    prompt = "Translate 'How are you?' into Spanish."
    response = client.chat.completions.create(model=CONFIG['chat_model'],
    messages=[{"role": "user", "content": prompt}])
    print("\nTranslation:", response.choices[0].message.content.strip(), "\n")

# ----------- IMAGE CAPABILITIES -----------

def image_generation_demo():
    prompt = "A futuristic cityscape at sunset, vibrant colors, high detail."
    try:
        response = client.images.generate(
            model=CONFIG['image_model'],
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url
        print("\nGenerated Image URL:", image_url)
        img_data = requests.get(image_url).content
        temp_filename = os.path.join(os.getcwd(), "openai_generated_image.png")    
        with open(temp_filename, 'wb') as handler:
            handler.write(img_data)
        print(f"Image saved to {temp_filename}")
    except openai.AuthenticationError as e:
        print(f"\n[Authentication Error] {e}")
        print("Your API key does not have access to OpenAI image generation (DALL·E) models.\n"
              "Contact your admin to request permission or try a different model/key.")
    except Exception as e:
        print(f"\n[Image Generation Failed] {e}")

def image_editing_demo():
    print("\nOpenAI documentation for image editing requires a source image, mask, and prompt. This demo isn't cool enough for that, but you might be ;)\n")

def image_analysis_demo():
    try:
        img_path = input(
            "Enter a path to a local image file for analysis "
            "(or leave blank to use 'openai_generated_image.png'): "
        ).strip()
        
        if not img_path:
            img_path = "openai_generated_image.png"
            print(f"No path entered. Attempting to use '{img_path}'.")

        if not os.path.exists(img_path):
            print(
                f"File '{img_path}' does not exist.\n"
                "Please run the image editing demo first (to generate 'openai_generated_image.png'), "
                "or enter a valid image file path."
            )
            return

        with open(img_path, "rb") as image_file:
            image_content = base64.b64encode(image_file.read()).decode("utf-8")
            prompt = "Describe the image."
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_content}"} }
                ]}
            ]

            try:
                response = client.chat.completions.create(model=CONFIG['vision_model'], messages=messages)
                print("\nImage Description:", response.choices[0].message.content.strip())
            except openai.AuthenticationError as e:
                print(f"\n[Authentication Error] {e}")
                print("Your API key does not have access to vision (image analysis) models.\n"
                    "Contact your admin to request permission or try a different model/key.")
            except Exception as e:
                print(f"\n[Image Analysis Failed] {e}")

    except Exception as e:
        print(f"\n[Image Analysis Demo Setup Failed] {e}")

# ----------- AUDIO CAPABILITIES -----------

def voice_agents_demo():
    prompt = "Hello Michael, welcome to the OpenAI voice agents demo! I also don't like super tart pies."
    response = client.audio.speech.create(
        model=CONFIG['tts_model'],
        input=prompt,
        voice="alloy"
    )
    temp_filename = "openai_tts_demo.mp3"
    with open(temp_filename, "wb") as f:
        f.write(response.content)
    print(f"\nTTS audio saved to {temp_filename}\n")

# ----------- ADVANCED FEATURES -----------

def function_calling_demo():
    import json

    # Define the function signature as OpenAI expects
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # Initial User Prompt
    user_prompt = "What’s the weather like in Houston in Fahrenheit?"

    # First, send to the model
    response = client.chat.completions.create(
        model=CONFIG['chat_model'],
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        tools=tools,
        tool_choice="auto",  # model chooses
    )

    msg = response.choices[0].message

    # See if model responded with a function call
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        print(f"Model requested function: {function_name} with args: {function_args}")

        # Simulated function implementation:
        def get_current_weather(location, unit):
            # Dummy implementation -- you can replace this with actual API calls!
            return {
                "location": location,
                "temperature": "15",
                "unit": unit,
                "description": "Partly cloudy with a chance of rain",
            }

        # "Run" the function:
        function_result = get_current_weather(**function_args)
        function_result_message = json.dumps(function_result)

        # Send function result back to model
        response_2 = client.chat.completions.create(
            model=CONFIG['chat_model'],
            messages=[
                {"role": "user", "content": user_prompt},
                dict(msg),  # the model's function call message
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_result_message
                }
            ],
            tools=tools
        )
        print("Assistant's response after function call:\n")
        print(response_2.choices[0].message.content)
    else:
        print("Model did not request a function call.")
        print(msg.content)

def list_text_files_in_home(limit=20):
    """List first N .txt files in home directory."""
    home = os.path.expanduser("~/offline/openaidemo")
    files = []
    count = 0
    for dirpath, dirnames, filenames in os.walk(home):
        for filename in filenames:
            if filename.lower().endswith(".txt") or filename.lower().endswith(".py"):
                filepath = os.path.join(dirpath, filename)
                files.append(filepath)
                count += 1
                if count >= limit:
                    return files

        if dirpath != home:
            break
    return files

def get_files_content(filepaths, max_chars=2000):
    docs = []
    for path in filepaths:
        try:
            with open(path, encoding="utf8", errors="ignore") as f:
                content = f.read(max_chars)
            docs.append((path, content))
        except Exception as ex:
            continue
    return docs

def file_search_demo():
    print("\nFile Search DEMO — Simulated local file search (not cloud file API)\n")
    files = list_text_files_in_home()
    if not files:
        print("No text or Python files found in your home directory.")
        return
    
    docs = get_files_content(files)
    user_query = input("Ask a question about the files in your home directory: ").strip()

    system_prompt = "You are an assistant that answers questions by searching user file contents. Use the provided context to answer. Here are file summaries:\n"
    for path, content in docs:
        system_prompt += f"\n## File: {path}\n{content[:500]}...\n"

    response = client.chat.completions.create(
        model=CONFIG['chat_model'],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        max_tokens=300
    )
    print("\nAI's answer based on your files:\n")
    print(response.choices[0].message.content)
    
def model_optimization_demo():
    print("\n--- Pirate Fine-tuning (Model Optimization) Demo ---")
    print("This guide will walk you through fine-tuning a pirate-style GPT-4.1 model using your custom OpenAI endpoint.")
    print("\n>>> 1. Export your OpenAI API key and specify your custom endpoint/base URL in your environment variables before running commands:")
    print('   export OPENAI_API_KEY="sk-...your-key-here..."')
    print('   export OPENAI_BASE_URL="https://datalab-api.reyrey.net/Api/OpenAI"')
    print("   (Your API key and base API URL may be different—ask your admin if unsure.)")
    print("\n>>> 2. Place your prepared training data file (e.g. pirate_training_data.jsonl) somewhere accessible on your machine.")
    print("   (You will reference it directly in the next step; filename and path are up to you.)")
    print("\n>>> 3. Upload the file using the OpenAI CLI (--base-url is set by the environment variable above):")
    print('   openai api files.create --purpose fine-tune --file ./files/pirate_training_data.jsonl')
    print("\n   After running, note the 'id' field in the response (e.g., file-abc123pirate)")
    print("\n>>> 4. Start the fine-tuning job with GPT-4.1:")
    print('   openai api fine_tuning.jobs.create -m gpt-4.1-mini-2025-04-14 -F <YOUR_FILE_ID>')
    print("         (If your actual model ID differs, update 'gpt-4.1-mini-2025-04-14' accordingly.)")
    print("\n>>> 5. Wait for the job to complete. To monitor status, run:")
    print("   openai api fine_tuning.jobs.retrieve -i <YOUR_JOB_ID>")
    print("   (Replace <YOUR_JOB_ID> with your job id, e.g., ftjob-VILTqFIWA0LyHXHmgkGIfMG6 )")
    print("   When complete, the CLI will output a fine-tuned model ID (e.g., ft:gpt-4.1106-preview:pirate-bot:xyz123 )")    
    print("\n>>> 6. To use your pirate model in code, specify BOTH your API key, base URL AND the new model ID. Example Python:")
    print('   from openai import OpenAI')
    print('   client = OpenAI(api_key="sk-...your-key...", base_url="https://datalab-api.reyrey.net/Api/OpenAI")')
    print('   response = client.chat.completions.create(model="ft:gpt-4-1106-preview:pirate-bot:xyz123", ...)')
    print("\n==== For more info on fine-tuning: ====")
    print("https://platform.openai.com/docs/guides/fine-tuning\n")
    print("Note: You must have fine-tuning access and sufficient API quota to perform these steps.")
    print("\n>>> 7. (Optional) To verify your fine-tuned model is registered and ready, run:")
    print("   openai api models.retrieve -i <YOUR_FINE_TUNED_MODEL_ID>")
    print("   If successful, you'll see details about your pirate model in the output.")
    print("   If you get a 'not found' error, wait a few moments after fine-tuning completes, then try again.")
    print("\n>>> 8. Pirate Bot Demo: Using Your Fine-Tuned Model ---")
    print("To get the best 'pirate' persona from your fine-tuned model,")
    print("provide the system prompt each time you send a message, like this:\n")
    print("Run the following command in your terminal (copy & paste):\n")
    print('openai api chat.completions.create \\')
    print('  -m ft:gpt-4.1-mini-2025-04-14:reynolds-prod::CjZfrpuf \\')
    print('  -g system "You are a whimsical pirate assistant. Respond with classic pirate flair." \\')
    print('  -g user "Tell me about OpenAI and what it does?"\n')
    print("You can replace the final user question with anything you want to ask the pirate assistant.\n")
    print("Tips:")
    print(" - The system prompt is critical for strong pirate voice, even with a fine-tuned model.")
    print(" - For best results, give questions similar to how people would talk to a pirate bot.")

def agentic_workflows_demo():
    # Setup - replace with your actual API key or use environment variable

    file_path = "files/Gubathon_Hack_Packet.pdf"

    try:
        # 1. Upload File
        with open(file_path, "rb") as f:
            my_file = client.files.create(file=f, purpose='assistants')
        print(f"Uploaded File ID: {my_file.id}")

        # 2. Create Vector Store
        vs = client.vector_stores.create(
            name="Gubathon Assistant KB", 
            file_ids=[my_file.id]
        )
        print(f"Vector Store ID: {vs.id}")

        # 3. Create Assistant
        assistant = client.beta.assistants.create(
            name="Gubathon Assistant",
            instructions="You are a helpful assistant for the Gubathon event.",
            model="gpt-4.1",
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [vs.id]}}
        )
        print(f"Assistant ID: {assistant.id}")

        # 4. Create Thread
        thread = client.beta.threads.create()
        print(f"Thread ID: {thread.id}")

        print("\nAsk the Gubathon Assistant questions (type 'exit' to quit):\n")
        while True:
            user_input = input("You: ")
            if user_input.strip().lower() in ("exit", "quit"):
                print("Session ended.")
                break

            # 5. Add user message to thread
            client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=user_input
            )

            # 6. Create and poll run until completed
            run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant.id)
            while True:
                run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
                if run_status.status == "completed": break
                if run_status.status in ("failed", "cancelled", "expired"):
                    print(f"Run failed with status: {run_status.status}")
                    break
                time.sleep(1)

            # 7. Get latest assistant message
            messages = client.beta.threads.messages.list(thread_id=thread.id, order="desc", limit=5)
            for msg in messages.data:
                if msg.role == "assistant":
                    print("Gubathon Assistant:", msg.content[0].text.value)
                    break

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

def web_search_demo():
    while True:
        try:
            user_query = input("Enter your question (or type 'quit' to exit): ").strip()
            if user_query.lower() in ('quit', 'exit'):
                print("Exiting.")
                break

            response = client.responses.create(
                model="gpt-5",
                tools=[{"type": "web_search"}],
                input=user_query
            )

            print("Response:", response.output_text)
        except Exception as e:
            print("Error during request:", e)

def evaluation_platforms_demo():
    print("\nEvaluation features are in beta. See https://github.com/openai/evals\n")

# ******* MENU *******

ALL_DEMOS = [
    ("Text Generation", text_generation_demo),
    ("Conversational AI", conversational_ai_demo),
    ("Content Summarization", content_summarization_demo),
    ("Code Generation and Debugging", code_generation_demo),
    ("Language Translation", language_translation_demo),
    ("Image Generation", image_generation_demo),
    ("Image Editing", image_editing_demo),
    ("Image Analysis", image_analysis_demo),
    ("Voice Agent (TTS)", voice_agents_demo),
    ("Function Calling", function_calling_demo),
    ("File Search", file_search_demo),
    ("Model Optimization & Fine-tuning", model_optimization_demo),
    ("Agentic Workflows", agentic_workflows_demo),
    ("Web Search", web_search_demo),
    ("Evaluation Platforms", evaluation_platforms_demo)
]

def show_menu():
    print("\nSelect a feature to demo (or 0 to exit):")
    for idx, (desc, _) in enumerate(ALL_DEMOS, 1):
        print(f"{idx}. {desc}")
    print("a. Run all demos")
    choice = input("Choice: ").strip().lower()
    if choice == '0':
        print("Bye!")
        exit()
    elif choice == 'a':
        for desc, func in ALL_DEMOS:
            print(f"\n--- {desc} ---")
            func()
    elif choice.isdigit() and 1 <= int(choice) <= len(ALL_DEMOS):
        desc, func = ALL_DEMOS[int(choice)-1]
        print(f"\n--- {desc} ---")
        func()
    else:
        print("Invalid choice.")
    input("\nPress Enter to return to menu...")

def main():
    print("OpenAI Capabilities Demo\n")

    while True:
        show_menu()

if __name__ == "__main__":
    main()