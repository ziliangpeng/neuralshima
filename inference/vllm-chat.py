from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging
import os

logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TQDM_DISABLE"] = "1"


def main():
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    llm = LLM(
        model="meta-llama/Llama-2-7b-chat-hf",
        disable_log_stats=True,  # hides progress bars / stats
        # disable_log_requests=True,     # hides per-request logs
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512, n=1)

    messages = [{"role": "system", "content": "You are helpful."}]

    print("Chat with the model! Type 'quit' to exit.")
    print("-" * 50)

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})

        # Generate prompt from conversation
        prompt = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        print(prompt)

        # Generate response
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        print(f"Assistant: {generated_text}")

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": generated_text})


if __name__ == "__main__":
    main()
