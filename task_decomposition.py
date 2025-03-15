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
lora_model_dir = "GPIoT_Task_Decomposition/checkpoint-13400"

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
model = AutoModelForCausalLM.from_pretrained(lora_model_dir)
model = model.to("cuda")
model.eval()

prompt = [
    {
        "role": "system",
        "content": "You are a professional IoT application developer. According to the user problem, you need to decompose the problem into multiple steps with implementation details. The output must be in the format of:\n\n1. description and implementation details of step 1\n\n2. description and implementation details of step 2\n\n......",
    },
    {
        "role": "user",
        "content": "Decompose the following task into multiple steps and describe the implementation details for each step. The task is to maximize throughput by digitally compensating for wireless impairments and removing residual interference from the transmit chain. ",
    },
]

input_text = data_transform(prompt)
input_ids = tokenizer(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=1024, num_return_sequences=1)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
