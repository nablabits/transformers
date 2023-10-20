
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


stop_sequence = " a company that"  # [257, 1664, 326]


def get_gpt2():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    return tokenizer, model


def generate_gpt2(prompt):
    tokenizer, model = get_gpt2()
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids
    stop_token_ids = tokenizer.encode(" a company  that")
    # outputs = model.generate(input_ids=input_ids)
    outputs = model.generate(input_ids=input_ids, eos_token_id=stop_token_ids)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def text2text_generator(prompt):
    text2text_generator = pipeline(
        "text2text-generation", model="gpt2", tokenizer="gpt2"
    )
    return text2text_generator(prompt, stop_sequence=" a company that")


if __name__ == "__main__":
    prompt = "Hugging Face Company is"
    # print(generate_tiny(prompt))
    # print(generate_gpt2(prompt))
    print(text2text_generator(prompt))


