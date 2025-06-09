import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
print(sys.executable)


# hyperparameters - adjusted for Python code
batch_size = 18
block_size = 64
max_iters = 10000
eval_interval = 500
learning_rate = 1e-3 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64           
n_head = 4           
n_layer = 4           
dropout = 0.1        
# ------------

torch.manual_seed(1337)

# Load the processed Python algorithms dataset
print("📚 Loading Python algorithms dataset...")
try:
    with open('python_algorithms_sample.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"✅ Dataset loaded: {len(text):,} characters")
except FileNotFoundError:
    print("❌ Dataset not found! Please run the data processor first.")
    print("Make sure 'python_algorithms_sample.txt' exists in your directory.")
    exit(1)

# Create vocabulary and mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔤 Vocabulary size: {vocab_size} characters")
print(f"📝 Sample characters: {chars[:20]}...")

# Create mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"📊 Training data: {len(train_data):,} tokens")
print(f"📊 Validation data: {len(val_data):,} tokens")

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 🧠 SELF-ATTENTION HEAD - The Core Innovation!
class Head(nn.Module):
    """One head of self-attention"""
    
    def __init__(self, head_size):
        super().__init__()
        # The three key transformations: Query, Key, Value
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)  
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Register the causal mask - prevents looking at future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels
        
        # Compute Query, Key, Value for all tokens
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)
        
        # Compute attention scores: "Who should I pay attention to?"
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        
        # Apply causal mask: can't look at future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Convert to probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Apply attention to values: "What information do I actually use?"
        out = wei @ v  # (B, T, head_size)
        return out
    
# 🔥 FEED FORWARD NETWORK - Non-linear Processing!
class FeedForward(nn.Module):
    """Simple feed-forward network"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand 4x
            nn.ReLU(),                       # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Project back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

 # 🏗️ TRANSFORMER BLOCK - The Complete Building Block!
class Block(nn.Module):
    """Transformer block: communication (attention) followed by computation (feedforward)"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)  # Self-attention
        self.ffwd = FeedForward(n_embd)                  # Feed-forward
        self.ln1 = nn.LayerNorm(n_embd)                  # Layer normalization
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections + layer norm (crucial for deep networks!)
        x = x + self.sa(self.ln1(x))      # Self-attention with residual
        x = x + self.ffwd(self.ln2(x))    # Feed-forward with residual
        return x
    
    
# 🎯 MULTI-HEAD ATTENTION - Multiple Perspectives!
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create multiple attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)  # Project back to original dimension
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel and concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back and apply dropout
        out = self.dropout(self.proj(out))
        return out


    
# 🚀 TRANSFORMER LANGUAGE MODEL - The Complete Architecture!
class TransformerLanguageModel(nn.Module):
    """Complete transformer model with self-attention"""
    
    def __init__(self):
        super().__init__()
        # Token and position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd) - Each token knows its position!
        
        # Pass through transformer blocks
        x = self.blocks(x)     # (B, T, n_embd)
        x = self.ln_f(x)       # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate new tokens using the trained model"""
        for _ in range(max_new_tokens):
            # Crop context to last block_size tokens
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, loss = self(idx_cond)
            
            # Focus on the last time step and apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Create the model
model = TransformerLanguageModel()
m = model.to(device)
print(f"🧠 Transformer model loaded on {device}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"🔢 Model parameters: {total_params:,}")
print(f"📊 Model size comparison:")
print(f"   - Bigram model: ~{vocab_size**2:,} parameters")
print(f"   - This Transformer: {total_params:,} parameters")
print(f"   - GPT-3: ~175,000,000,000 parameters!")

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\n🚀 Starting training...")
print("=" * 50)

# Training loop
for iter in range(max_iters):
    # Evaluate occasionally
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Generate a sample to see progress
        if iter > 0:
            print("\n📝 Sample generation:")
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            sample = decode(m.generate(context, max_new_tokens=200, temperature=0.8)[0].tolist())
            print("-" * 30)
            print(sample[:200] + "...")
            print("-" * 30)

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("\n✅ Training complete!")

# Enhanced generation functions
def generate_code(prompt="", max_length=300, temperature=0.7):
    """Generate Python code with optional prompt"""
    if prompt:
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    generated = m.generate(context, max_new_tokens=max_length, temperature=temperature)
    return decode(generated[0].tolist())

def complete_function(function_start, max_length=200):
    """Complete a Python function given its beginning"""
    return generate_code(function_start, max_length, temperature=0.8)

def generate_algorithm(algorithm_type="sort", max_length=250):
    """Generate an algorithm of a specific type"""
    prompts = {
        "sort": "def ",
        "search": "def binary_search(",
        "math": "def calculate_",
        "list": "def process_list(",
        "string": "def string_"
    }
    
    prompt = prompts.get(algorithm_type, "def ")
    return generate_code(prompt, max_length, temperature=0.7)

# Test the trained model
print("\n🎯 Testing the Self-Attention Python Code Generator:")
print("=" * 50)

# Test 1: Free generation
print("\n1. Free code generation:")
sample1 = generate_code(max_length=300)
print(sample1)

# Test 2: Function completion
print("\n2. Function completion:")
function_start = "def fibonacci(n):\n    if n <= 1:\n        return"
completion = complete_function(function_start)
print(completion)

# Test 3: Algorithm generation
print("\n3. Algorithm generation:")
algorithm = generate_algorithm("sort")
print(algorithm)

# Test 4: Interactive generation
print("\n4. 🎪 Interactive generation - try different prompts!")
test_prompts = [
    "def bubble_sort(",
    "def is_prime(",
    "class LinkedList:",
    "for i in range("
]

for prompt in test_prompts:
    print(f"\n🎯 Prompt: '{prompt}'")
    result = generate_code(prompt, max_length=150, temperature=0.6)
    print(f"Generated: {result[:100]}...")

print("\n🌟 Amazing! Your model now has MEMORY and CONTEXT AWARENESS!")
print("🧠 It can remember function names, variable types, and coding patterns!")
print("🚀 This is the same architecture that powers ChatGPT and GPT-4!")

# Save the model
print("\n💾 Saving model...")
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
}, 'python_code_generator.pth')

print("✅ Model saved as 'python_code_generator.pth'")

