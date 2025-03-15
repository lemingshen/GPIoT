from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


def data_transform(message_list):
    system_message = message_list[0]["content"]
    user_message = message_list[1]["content"]

    output = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{}[/INST]".format(
        system_message, user_message.strip()
    )

    return output


base_model_dir = "meta-llama/Llama-2-13b-chat-hf"
lora_model_dir = "GPIoT_Code_Generation/checkpoint-13000"

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
model = AutoModelForCausalLM.from_pretrained(lora_model_dir)
model = model.to("cuda")
model.eval()

prompt = [
    {
        "role": "system",
        "content": "You are a professional and skillful Python programmer, especially in the field of communication, signal processing, and machine learning. According to the user instruction, you need to generate one single Python function with detailed comments and documentation. The documentation should be in the Markdown format.",
    },
    {
        "role": "user",
        "content": "**Target**\nDefine a Python function to create a simple augmentation pipeline for image processing and provide detailed code comments.\n\n**Input Specifications**\n- `image_path` (str): The file path to the input image.\n\n**Output specifications**\nThe function does not explicitly return any value but visualizes the original and augmented images using matplotlib.",
    },
]

input_text = data_transform(prompt)
input_ids = tokenizer(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=1024, num_return_sequences=1)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
