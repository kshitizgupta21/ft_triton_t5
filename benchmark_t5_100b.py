import argparse
import time
import torch
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size')
parser.add_argument('--input-seq-len', type=int, default=64,
                    help='batch size')
parser.add_argument('--output-seq-len', type=int, default=32,
                    help='output-len')

# example: python3 benchmark_t5_100b.py --batch-size 1 --input-seq-len 64 --output-seq-len 32

args = parser.parse_args()  

def prepare_tensor(name, input):
    t = client_util.InferInput(
        name, input.shape, np_to_triton_dtype(input.dtype))
    t.set_data_from_numpy(input)
    return t

client_util = httpclient
client = client_util.InferenceServerClient(url = "localhost:8000", concurrency=1)

class InputTokens(object):
    def __init__(self, batch_size, input_seq_len, bos_token, eos_token, vocab_size):
        # Set the last token of each sequence to eos and replace the bos/eos tokens in the middle of the sequences to
        # some other tokens.
        normal_token_list = list(range(vocab_size))
        if bos_token in normal_token_list:
            normal_token_list.remove(bos_token)
        if eos_token in normal_token_list:
            normal_token_list.remove(eos_token)
        self.input_ids = torch.randint(0, len(normal_token_list), (batch_size, input_seq_len))
        for batch_idx in range(batch_size):
            for token_idx in range(input_seq_len):
                if token_idx == input_seq_len - 1:
                    self.input_ids[batch_idx][token_idx] = eos_token
                else:
                    self.input_ids[batch_idx][token_idx] = normal_token_list[self.input_ids[batch_idx][token_idx]]
        # Set attention masks to all ones.
        self.attention_mask = torch.ones((batch_size, input_seq_len), dtype=torch.int64)

decoder_start_token_id = 0
eos_token_id = 1
vocab_size = 32128
input_token = InputTokens(args.batch_size, args.input_seq_len, decoder_start_token_id, eos_token_id, vocab_size)

input_ids = input_token.input_ids.numpy().astype(np.uint32)
mem_seq_len = torch.sum(input_token.attention_mask, dim=1).numpy().astype(np.uint32)
mem_seq_len = mem_seq_len.reshape([mem_seq_len.shape[0], 1])
max_output_len = (args.output_seq_len * np.ones([input_ids.shape[0], 1])).astype(np.uint32)
start_ids = decoder_start_token_id * np.ones([input_ids.shape[0], 1]).astype(np.uint32)

# Set eos token id to -1 to effectively disable early stopping.
end_ids = -1 * np.ones([input_ids.shape[0], 1]).astype(np.int64)


inputs = [
    prepare_tensor("input_ids", input_ids),
    prepare_tensor("sequence_length", mem_seq_len),
    prepare_tensor("max_output_len", max_output_len),
    prepare_tensor("start_id", start_ids),
    prepare_tensor("end_id", end_ids),
]

model_name = "fastertransformer"
result = client.infer(model_name, inputs)

ft_decoding_outputs = result.as_numpy("output_ids")
ft_decoding_seq_lens = result.as_numpy("sequence_length")
print("ft_decoding_outputs:", ft_decoding_outputs)
print("ft_decoding_seq_lens:", ft_decoding_seq_lens)

# warmup
for _ in range(5):
    result = client.infer(model_name, inputs)
latencies = []
start_tot_time = time.time()
iters = 10
for _ in range(iters):
    start_time = time.time()
    result = client.infer(model_name, inputs)
    latency = (time.time() - start_time) * 1000
    latencies.append(latency)
end_tot_time = time.time()
tot_time = end_tot_time - start_tot_time
latencies = np.array(latencies)

print(f"P50 latency is {np.percentile(latencies, 50)}ms")
print(f"Throughput is {(iters * args.batch_size * args.output_seq_len)/tot_time} tokens/sec")

