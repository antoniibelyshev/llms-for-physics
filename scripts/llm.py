def sample_physics(k_best, model, tokenizer, prompt_head) -> str:
    prompt = prompt_head + '\n\n'.join(k_best)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=100,  # Maximum length of the generated response
        num_return_sequences=1,  # Number of responses to generate
        # no_repeat_ngram_size=2,  # Prevent repeating n-grams
        early_stopping=True,  # Stop the generation early if EOS token is reached
        top_k=50,  # Top-K sampling
        top_p=0.95,  # Top-p (nucleus) sampling
        temperature=0.7,  # Sampling temperature
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
