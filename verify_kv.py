import torch
import time
import matplotlib.pyplot as plt 
import tiktoken
from model import GPT

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
if torch.backends.mps.is_available(): device = 'mps'
print(f"Running on device {device}")

# Load Model (optimized with KV Cache changes)
model = GPT.from_pretrained('gpt2', dict(dropout=0.0))
model.eval()

# Force Flash=False
for module in model.modules():
	if hasattr(module, 'flash'):
		module.flash = False
		# Ensure bias buffer is registered
		if not hasattr(module, 'bias'):
			config = model.config
			bias = torch.tril(torch.ones(1, 1, config.block_size, config.block_size))
			module.register_buffer("bias", bias)
model.to(device)
# Setup Tokeniser
enc = tiktoken.get_encoding("gpt2")
prompt = "What is Life?"
x = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, ...]

# Per token latency test
print("Generating tokens and measuring latency per step (Expecting every step to take the same amount of time)")
latencies = []
generated_ids = []
# Run the loop to measure distinct steps
past_kv = None 
max_new_tokens = 200

with torch.no_grad():
	for i in range(max_new_tokens):
		start = time.time()
		# Prepare input
		if past_kv is None:
			idx_cond = x 
		else:
			idx_cond = x[:, -1:] # Feed only the last token

		# Forward Pass
		logits, _, past_kv = model(idx_cond, cache=past_kv)

		if device == 'cuda':
			torch.cuda.synchronize()
		if device == 'mps':
			torch.mps.synchronize()
		end = time.time()
		latencies.append((end - start) * 1000)

		# Greedy sampling - Pick the logit with the highest probability
		logits = logits[:, -1, :]
		idx_next = torch.argmax(logits, dim=-1, keepdim=True)
		# Append sampled token
		x = torch.cat((x, idx_next), dim=1)
		generated_ids.append(idx_next.item())

# Decode Output
output_text = enc.decode(generated_ids)
print(f"Generated Text : {prompt} {output_text}")

plt.figure(figsize=(10, 5))
plt.plot(latencies, marker='o', linestyle='-', color='b', alpha=0.5)
plt.title(f"Inference Time per Token on {device}")
plt.xlabel("Token Index (0 to 200)")
plt.ylabel("Time")
plt.grid(True)
plt.savefig(f"kv_cache_verification_{device}.png")
plt.show()