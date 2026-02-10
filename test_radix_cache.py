import torch
import time
import matplotlib.pyplot as plt
import tiktoken
import copy
from model import GPT
from radix_tree import RadixCache, RadixNode

device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
elif torch.backends.mps.is_available(): device = 'mps'
print(f"Running on {device}")

print("Loading Model")
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()

for module in model.modules():
	if hasattr(module, 'flash'):
		module.flash = False
		# Ensure bias buffer is registered
		if not hasattr(module, 'bias'):
			config = model.config
			bias = torch.tril(torch.ones(1, 1, config.block_size, config.block_size))
			module.register_buffer("bias", bias)

model.to(device)
enc = tiktoken.get_encoding("gpt2")
radix_tree = RadixCache()

prompts = [
    "The recipe for a perfect chocolate cake involves",                  # Prompt A
    "The recipe for a perfect chocolate cake involves adding vanilla",   # Prompt B (Prefix Match)
    "The recipe for a perfect chocolate cake involves adding vanilla",   # Prompt C (Exact Match)
    "The recipe for a perfect chocolate cake involves adding chocolate",
    "The recipe for a perfect chocolate cake involves adding strawberry",
    "The recipe for a perfect chocolate cake involves adding chocolate",
    "The recipe for a perfect chocolate cake involves adding vanilla",
    "Quantum mechanics describes the behavior of"                        # Prompt D (No Match)
]

results = {'prompt_id': [], 'match_len': [], 'time': [], 'status': []}

print("\n--- Starting Radix Cache Test ---")

for i, text in enumerate(prompts):
    print(f"\nPrompt {i+1}: \"{text[:40]}...\"")
    
    # 1. Encode
    tokens = torch.tensor(enc.encode(text), dtype=torch.long, device=device)
    
    # 2. Check Cache
    cached_kv, match_len = radix_tree.match_prefix(tokens)
    
    # 3. Prepare Input (Suffix)
    if match_len > 0:
        x_input = tokens[match_len:][None, ...] # Suffix
        status = "PARTIAL HIT" if match_len < len(tokens) else "FULL HIT"
    else:
        x_input = tokens[None, ...] # Full prompt
        status = "MISS"
        
    print(f"  [Cache] Status: {status} | Matched Tokens: {match_len}/{len(tokens)}")

    # 4. Measure Pre-fill Time
    # (We flush GPU/MPS to get accurate timing)
    if device == 'cuda': torch.cuda.synchronize()
    elif device == 'mps': torch.mps.synchronize()
    
    t0 = time.time()
    
    with torch.no_grad():
        # Handle the edge case where Full Match means suffix is empty
        if x_input.shape[1] == 0:
            # If full match, we already have the KV! 
            # We don't run model() on empty input. We just use cached_kv.
            new_kv = cached_kv
        else:
            # Run model on suffix (or full prompt)
            # IMPORTANT: We assume model.forward accepts past_kv
            _, _, new_kv = model(x_input, cache=cached_kv)
            
    if device == 'cuda': torch.cuda.synchronize()
    elif device == 'mps': torch.mps.synchronize()
    t1 = time.time()
    
    latency = (t1 - t0) * 1000 # ms
    print(f"  [Perf]  Pre-fill Latency: {latency:.2f} ms")
    
    # 5. Insert into Tree
    # Note: We must insert the FULL tokens and FULL KV cache
    # If we had a partial match, 'new_kv' is already the full cache because 
    # the model appended the new tokens to the past_kv.
    radix_tree.insert(tokens, new_kv)
    
    # Log results
    results['prompt_id'].append(f"P{i+1}")
    results['match_len'].append(match_len)
    results['time'].append(latency)
    results['status'].append(status)

# ==========================================
# 4. Plotting
# ==========================================
plt.figure(figsize=(10, 6))
bars = plt.bar(results['prompt_id'], results['time'], color=['red', 'orange', 'green', 'red'])

# Add labels
for bar, status, match in zip(bars, results['status'], results['match_len']):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, status, ha='center', va='bottom', fontweight='bold')
    plt.text(bar.get_x() + bar.get_width()/2, yval/2, f"Match: {match}", ha='center', va='center', color='white')

plt.title("KV Cache Pre-fill Latency (Radix Tree)")
plt.ylabel("Time (ms)")
plt.xlabel("Prompt Sequence")
plt.grid(axis='y', alpha=0.3)
plt.savefig("radix_cache_results.png")
print("\nTest Complete. Plot saved to 'radix_cache_results.png'")
plt.show()

def run_test(prompt_text, tokens_to_generate=15):
    print(f"\nPrompt: \"{prompt_text}\"")
    tokens = torch.tensor(enc.encode(prompt_text), dtype=torch.long, device=device)
    
    # --- Step A: Check Radix Tree ---
    cached_kv, match_len = radix_tree.match_prefix(tokens)
    
    # --- Step B: Logic for Full/Partial Matches ---
    # If we matched the WHOLE prompt, we must 'rewind' by 1 token 
    # so the model has something to process to generate the NEXT token.
    if match_len == len(tokens):
        print(f"   [Status] FULL HIT (Rewinding 1 token)")
        match_len -= 1
        cached_kv = radix_tree._slice_kv_cache(cached_kv, match_len)
    elif match_len > 0:
        print(f"   [Status] PARTIAL HIT (Matched {match_len} tokens)")
    else:
        print(f"   [Status] MISS (Cold Start)")

    # Define the suffix (input to the model)
    x_input = tokens[match_len:][None, ...]

    # --- Step C: Pre-fill (and First Token Gen) ---
    t0 = time.time()
    with torch.no_grad():
        logits, _, past_kv = model(x_input, cache=cached_kv)
    t1 = time.time()
    prefill_time = (t1 - t0) * 1000

    # --- Step D: Generation Loop ---
    generated_tokens = []
    curr_kv = past_kv
    # Get first predicted token from the pre-fill logits
    next_token_logits = logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_tokens.append(next_token.item())
    
    # Generate remaining tokens
    for _ in range(tokens_to_generate - 1):
        with torch.no_grad():
            logits, _, curr_kv = model(next_token, cache=curr_kv)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())

    # --- Step E: Insert into Tree ---
    # We insert the prompt + generated text so future queries can reuse it all
    full_sequence = torch.cat((tokens, torch.tensor(generated_tokens, device=device)))
    radix_tree.insert(full_sequence, curr_kv)

    # Output
    gen_text = enc.decode(generated_tokens)
    print(f"   [Output] ...{gen_text}")
    print(f"   [Time]   Pre-fill: {prefill_time:.2f} ms")

# ==========================================
# 4. Scenarios
# ==========================================

# 1. Cold Start
run_test("The quick brown fox jumps over the")

# 2. Shared Prefix (Should be faster)
# Shares "The quick brown fox jumps over "
run_test("The quick brown fox jumps over the lazy dog") 

# 3. Exact Repeat (Should be fastest, uses rewind logic)
run_test("The quick brown fox jumps over the lazy dog")