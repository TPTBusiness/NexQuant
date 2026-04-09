#!/bin/bash
# Start llama.cpp server with Qwen3.5-35B for strategy generation
# Usage: ./start_llama.sh

MODEL_PATH="$HOME/models/qwen3.5/Qwen3.5-35B-A3B-Q3_K_M.gguf"
PORT=8081

# GPU: RTX 5060 Ti (16GB VRAM)
# Modell braucht ~15.7GB bei voller GPU-Nutzung
# Mit 14GB free → reduzieren wir GPU-Layers + Context
GPU_LAYERS=30  # Weniger Layers für VRAM
CTX_SIZE=4096  # 4K Context (reicht für Strategien)

echo "🚀 Starting llama.cpp server..."
echo "   Model: $(basename $MODEL_PATH)"
echo "   Port: $PORT"
echo "   GPU Layers: $GPU_LAYERS"
echo "   Context: $CTX_SIZE"
echo ""

exec ~/llama.cpp/build/bin/llama-server \
  --model "$MODEL_PATH" \
  --n-gpu-layers $GPU_LAYERS \
  --ctx-size $CTX_SIZE \
  --port $PORT \
  --threads 8 \
  --threads-batch 8 \
  --parallel 1 \
  --flash-attn \
  --jinja \
  --host 0.0.0.0
