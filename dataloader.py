import torch
import numpy as np
import os
import tiktoken


# def load_tokens(filename):
#     npt = np.load(filename)
#     ptt = torch.tensor(npt, dtype = torch.long)
#     return ptt

# class DataLoader:
#     def __init__(self, B, T, process_rank, num_processes, split, master_process = False):
#         self.B = int(B)
#         self.T = int(T)
#         self.process_rank = process_rank
#         self.num_processes = num_processes
#         assert split in {'train', 'val'}

#         dataset_root = '/content/drive/MyDrive/dataset_for_GPT_70M'
#         shards = os.listdir(dataset_root)
#         shards = [s for s in shards if split in s]
#         shards = sorted(shards)
#         shards = [os.path.join(dataset_root, s) for s in shards]

#         self.shards = shards

#         assert len(shards) > 0, f'no shards found for split {split}'

#         if master_process:
#             print(f'found {len(shards)} shards for split {split}')

#         self.reset()

#     def reset(self):
#         self.current_shard = 0
#         self.tokens = load_tokens(self.shards[self.current_shard])
#         self.current_position = self.B * self.T * self.process_rank

#     def next_batch(self):
#         B, T = self.B, self.T
#         buffer = self.tokens[self.current_position : self.current_position + self.B * self.T + 1]

#         x = buffer[:-1].view(B, T)
#         y = buffer[1:].view(B, T)

#         self.current_position += B * T * self.num_processes

#         if (self.current_position + B * T * self.num_processes) >= len(self.tokens):
#             self.current_shard = (self.current_shard + 1) % len(self.shards)
#             self.tokens = load_tokens(self.shards[self.current_shard])
#             self.current_position = B * T * self.process_rank

#         return x, y


class DataLoader:
    def __init__(self, B, T, process_rank, num_processes, master_process = False):
        self.B = int(B)
        self.T = int(T)
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding('p50k_base')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        if master_process:
            print(f"loaded {len(self.tokens)} tokens")
            temp = len(self.tokens) // (B * T)
            print(f"1 epoch = {temp} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y
