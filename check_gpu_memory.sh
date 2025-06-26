#!/usr/bin/env bash
# Quick utility to display GPU memory summary (CSV format).
# Usage:  ./check_gpu_memory.sh
# Note: nvidia-smi shows raw free VRAM. PyTorch may still OOM if it cannot
#       find a single contiguous block. Example error observed:
#       torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 52.33 GiB. GPU 0 
#       has a total capacity of 79.15 GiB of which 23.34 GiB is free. Including non-PyTorch memory, 
#       this process has 55.80 GiB memory in use. Of the allocated memory 49.44 GiB is allocated by 
#       PyTorch, and 5.85 GiB is reserved by PyTorch but unallocated. If reserved but unallocated 
#       memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid 
#       fragmentation. 

# Mitigations • Reduce --pipeline.datamanager.train-num-rays-per-batch (halve and retry).
# • Or keep 16 k rays/batch but set
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:True

# max_split_size_mb:512
# Tells the allocator that if it cannot find a single contiguous block larger than 512 MB, it is allowed 
# to split a very large tensor request into smaller 512 MB-sized chunks.

# expandable_segments:True
# Lets the allocator grow an existing memory segment in place instead of always grabbing a brand-new one, 
# further reducing fragmentation.

# Together they relax the contiguous-block requirement that caused the OOM you saw, letting PyTorch 
# satisfy big allocations even when free VRAM is fragmented. It doesn’t reduce peak memory usage, but it 
# makes “I have free memory yet still OOM” situations far less likely.
# Then launch training, e.g.:
#   PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF ./run_difix3d_train.sh

nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
