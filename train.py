import time
import math
import os
import torch
from torch.nn import functional as F

from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import tiktoken

from model import GPT
from dataloader import DataLoader


# Set up connection with drive
from google.colab import drive
drive.mount('/content/drive')


if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

torch.set_float32_matmul_precision('high')


################## Change for A100(bfloat16) or T4(float16) ####################
torch_dtype = torch.float16
################################################################################

# Setting up DDP
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), " look in hyperparameters section if cuda is available"
    init_process_group(backend = 'nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device(f'cuda:{ddp_local_rank}')
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    device = "cpu"
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        
    elif torch.cuda.is_available():
        device = "cuda"

    print(f"Using device {device}")


# Model hyperparameters
n_embed = 448           # Size of embeddings
n_head = 8              # Number of attention heads
n_layer = 10            # Number of transformer layers
batch_size = 16         # Batch size
block_size = 1024       # Context length
dropout = 0.1           # Dropout probability
vocab_size = 100352     # Vocabulary size :: Originally = 100277, update to nice number 100352 -> divisible by 2048

warmup_steps = 222  # calculate by:: max_steps * (715 / 19073)
max_learning_rate = 6e-4
min_learning_rate = 0.1 * max_learning_rate
max_steps = 5912  # 5912 steps -> 3.1B training tokens, change if amount of tokens change
                  # Additional 300M tokens for validation loss

val_interval = 100
generate_interval = 200
checkpoint_interval = 200

class Hyperparameters:
    vocab_size = vocab_size
    n_head = n_head
    n_embed = n_embed
    n_layer = n_layer
    block_size = block_size

hyperparameters = Hyperparameters()

model = GPT(hyperparameters, master_process = True)
model = model.to(device = device)

############################ Set True in A100 ##################################
use_compile = False
if use_compile:
    model = torch.compile(model)
################################################################################

if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])

# List all the parameters of the model
if master_process:
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_params = count_parameters(model)
    print(f"The model has {num_params} trainable parameters.")

    def count_parameters_by_layer(model):
        total_params = 0
        print("Layer-wise parameter count:")

        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                print(f"Layer: {name}, Parameters: {num_params}")
                total_params += num_params

        print(f"Total Trainable Parameters: {total_params}")

    count_parameters_by_layer(model)


raw_model = model.module if ddp else model
device_type = "cuda" if device.startswith("cuda") else "mps" if device.startswith("mps") else "cpu"

enc = tiktoken.get_encoding("cl100k_base")

B = batch_size
T = block_size

# In order to simulate GPT-3 0.5M batch size
total_batch_size = 524288 # (2 ^ 19) -> ~0.5M batch size
assert total_batch_size % (B * T * ddp_world_size) == 0, "check total_batch_size portion again"
grad_accumulation_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Gradient accumulation steps: {grad_accumulation_steps}")
    print(f"=> calculated gradient accumulation steps = {grad_accumulation_steps}")

train_loader = DataLoader(
    B = B,
    T = T,
    process_rank = ddp_rank,
    num_processes = ddp_world_size,
    split = 'train',
    master_process = True
  )

val_loader = DataLoader(
    B = B,
    T = T,
    process_rank = ddp_rank,
    num_processes = ddp_world_size,
    split = 'val',
    master_process = True
  )

# Learning rate cosine variation
def get_learning_rate(it):
    if it < warmup_steps:
        return max_learning_rate * (it + 1) / (warmup_steps)

    if it > max_steps:
        return min_learning_rate

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, f"Decay ratio out of range: {decay_ratio}"

    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


# Optimizer
optimizer = raw_model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-4, device_type = device_type)


# Set up directories and files for log and dataset
# drive_base_dir = './'
drive_base_dir = '/content/drive/MyDrive/GPT_70M_saves/'

log_dir = os.path.join(drive_base_dir, 'checkpoints')
os.makedirs(log_dir, exist_ok = True)

log_file = os.path.join(log_dir, f"log.txt")
if not os.path.exists(log_file): # if not exist, then clear start
    with open(log_file, "w") as f:
        pass

# Remove very old checkpoints to save drive space
import glob

def clean_old_checkpoints(directory, keep = 3):
    ckpts = glob.glob(os.path.join(directory, "checkpoint_step*.pt"))
    ckpts.sort(key = lambda x: int(x.split("checkpoint_step")[1].split(".pt")[0]))
    for ckpt in ckpts[:-keep]:
        os.remove(ckpt)

# If resuming training, then load from checkpoint
# checkpoint_path = '/content/drive/MyDrive/GPT_Tokenizer_saves/checkpoints/checkpoint_step1.pt'
checkpoint_path = None

if checkpoint_path:
    if master_process:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(log_file, "a") as f:
            f.write(f"\nResuming from checkpoint: {checkpoint_path}\n")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    raw_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    step_offset = checkpoint.get('step', 0) + 1

    # Safely load torch RNG state
    torch_rng = checkpoint['torch_rng_state']
    if not isinstance(torch_rng, torch.ByteTensor):
        torch_rng = torch_rng.cpu().clone().detach().type(torch.ByteTensor)
    torch.set_rng_state(torch_rng)

    # Safely load CUDA RNG states
    if device_type == "cuda":
        cuda_rng_state = checkpoint['cuda_rng_state']
        cuda_rng_state = [
            s.cpu().clone().detach().type(torch.ByteTensor)
            if not isinstance(s, torch.ByteTensor) else s.cpu()
            for s in cuda_rng_state
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state)

else:
    step_offset = 0
    if master_process:
        print("Starting fresh training loop")
        with open(log_file, "a") as f:
            f.write("\nStarting fresh training loop\n")



time_tracker = []

# Training loop
for step in range(step_offset, max_steps):
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % val_interval == 0 or last_step:
        time_validation_loss_start = time.time()

        model.eval()
        val_loader.reset()

        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 310

            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)

                with torch.autocast(device_type = device_type, dtype = torch_dtype):
                    logits, loss = model(x, y)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()

        if ddp:
            dist.all_reduce(val_loss_accum, op = dist.ReduceOp.AVG)

        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} validation loss = {val_loss_accum.item():.4f}\n")

            time_validation_loss = time.time() - time_validation_loss_start ;
            print("Validation loss time =", time_validation_loss)

            time_checkpointing_start = time.time()

            if step > 0 and (step % checkpoint_interval == 0 or last_step):
                checkpoint = {
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                    'torch_rng_state': torch.get_rng_state().cpu().clone().detach().type(torch.uint8),
                    'cuda_rng_state': [s.clone().detach().cpu().type(torch.uint8) for s in torch.cuda.get_rng_state_all()],
                    'grad_accumulation_steps': grad_accumulation_steps,
                    'batch_size': batch_size,
                    'block_size': block_size,
                    'learning_rate': learning_rate,
                    'max_steps': max_steps,
                    'manual_lr_schedule': {
                        'warmup_steps': warmup_steps,
                        'max_lr': max_learning_rate,
                        'min_lr': min_learning_rate
                    }
                }

                torch.save(checkpoint, os.path.join(log_dir, f"checkpoint_step{step}.pt"))
                clean_old_checkpoints(log_dir)

                time_checkpointing = time.time() - time_checkpointing_start
                print("Checkpointing time =", time_checkpointing)



    # # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % generate_interval == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 200

        tokens = enc.encode("while(True):")
        tokens = torch.tensor(tokens, dtype = torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device = device)
        sample_rng.manual_seed(42 + ddp_rank)

        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type = device_type, dtype = torch_dtype):
                    logits, loss = model(xgen) # (B, T, vocab_size)

                logits = logits[:, -1, :] # (B, vocab_size)
                probs = F.softmax(logits, dim = -1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim = -1)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                xgen = torch.cat((xgen, xcol), dim=1)

        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    time_training_start = time.time()

    model.train()
    optimizer.zero_grad(set_to_none = True)
    loss_accum = 0.0

    for micro_step in range(grad_accumulation_steps): # to simulate large parallel batch size in gpt(0.5 Million)
        x_batch, y_batch = train_loader.next_batch()
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)

        with torch.autocast(device_type= device_type, dtype = torch_dtype):
            logits, loss = model(x_batch, y_batch)

        loss = loss / grad_accumulation_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op = dist.ReduceOp.AVG)

    # gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    learning_rate = get_learning_rate(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize()

    time_training = time.time() - time_training_start # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accumulation_steps * ddp_world_size
    tokens_per_sec = tokens_processed // time_training


    time_tracker.append(time_training)
    if len(time_tracker) > 10:
        time_tracker.pop(0)  # keep last 10 steps

    avg_dt = sum(time_tracker) / len(time_tracker)
    eta_min = avg_dt * (max_steps - step - 1) / 60
    eta_hours = int(eta_min // 60)
    eta_minutes = int(eta_min % 60)

    if master_process:
        log_string = (f"Step {step}, Loss: {loss_accum.item():.6f} | LR = {learning_rate:.4e} | "
                  f"GradNorm = {norm:.4f} | Time: {time_training:.2f}s | tok/sec = {tokens_per_sec} | ETA: {eta_hours}hrs {eta_minutes}min")

        print(log_string, flush = True)

        with open(log_file, "a") as f:
            f.write(log_string + "\n")


if master_process:
    print("Training finished. Saving final model checkpoint...")
    final_checkpoint_path = os.path.join(log_dir, "final_model.pt")

    final_checkpoint = {
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        'val_loss': val_loss_accum.item(),
        'torch_rng_state': torch.get_rng_state().cpu().clone().detach().type(torch.uint8),
        'cuda_rng_state': [s.clone().detach().cpu().type(torch.uint8) for s in torch.cuda.get_rng_state_all()],
        'grad_accumulation_steps': grad_accumulation_steps,
        'batch_size': batch_size,
        'block_size': block_size,
        'learning_rate': learning_rate,
        'max_steps': max_steps,
        'manual_lr_schedule': {
            'warmup_steps': warmup_steps,
            'max_lr': max_learning_rate,
            'min_lr': min_learning_rate
        }
    }

    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")


if ddp:
    destroy_process_group()

