import dspy
import os

# 1. Set your Zhipu API Key
api_key = "7bc9d8e4b5444b97b5a25e071c478e2c.P7iDRHY4rhaDZYQk"
os.environ["OPENAI_API_KEY"] = api_key

print("Connecting to Zhipu AI server...")

# 2. Connect to the domestic model using OpenAI compatibility mode
# CRITICAL FIX: Added `model_type='chat'` because the model name doesn't contain 'gpt'.
# Without this, DSPy defaults to the wrong (deprecated) endpoint format.
my_domestic_model = dspy.OpenAI(
    model='glm-4-flash',
    api_key=api_key,
    api_base='https://open.bigmodel.cn/api/paas/v4/',
    model_type='chat',  # <--- THE MAGIC FIX IS HERE
    max_tokens=250
)

# Configure DSPy to use this model by default
dspy.settings.configure(lm=my_domestic_model)


# 3. Define a simple DSPy module
class HelloWorld(dspy.Signature):
    """Answer the user's simple question."""
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Generated answer")


# 4. Execute the prediction and print the result
try:
    predictor = dspy.Predict(HelloWorld)
    # Give it a small test
    response = predictor(
        question="Hello! Please greet me in one short sentence and tell me which model version you are currently using.")

    print("\n🎉 Connection successful! The model's response is as follows:")
    print("-" * 30)
    print(response.answer)
    print("-" * 30)
except Exception as e:
    print("\n❌ Oops, connection failed. Error message:")
    print(e)