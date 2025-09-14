import openai
import os

def main():
    # Set up the client with Perplexity API
    client = openai.OpenAI(
        api_key=os.getenv("PPLX_API_KEY"),
        base_url="https://api.perplexity.ai"
    )
    
    # Initialize conversation history
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    print("Chat with Perplexity! Type 'quit' to exit.")
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
        
        try:
            # Generate response using Perplexity API
            response = client.chat.completions.create(
                # model="llama-3.1-sonar-small-128k-online",
                model="sonar",
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            
            assistant_message = response.choices[0].message.content
            print(f"Assistant: {assistant_message}")
            
            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": assistant_message})
            
        except Exception as e:
            print(f"Error: {e}")
            # Remove the user message if there was an error
            messages.pop()

if __name__ == "__main__":
    main()
