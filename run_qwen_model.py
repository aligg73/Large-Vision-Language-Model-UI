import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
import torch
import argparse
import os

# Use a relative path or environment variable for the model path
model_path = os.environ.get("QWEN_MODEL_PATH", "path/to/model")

def load_model(use_flash_attention=False):
    num_gpus = torch.cuda.device_count()
    
    # Load config first to get number of layers
    config = AutoConfig.from_pretrained(model_path)
    total_layers = config.num_hidden_layers
    
    if num_gpus <= 1:
        device_map = {"": 0}
    else:
        device_map = {
            # Vision components on GPU 1
            "vision_model": 1,
            "vision_projection": 1,
            "model.embed_tokens": 0,
            "perceiver": 1,
            "image_processor": 1,
            "rotary_emb": 0,
            "model.rotary_emb": 0,
            
            # Visual merger components
            "visual.merger": 1,
            "visual.merger.ln_q": 1,
            "visual.merger.ln_k": 1,
            "visual.merger.ln_v": 1,
            "visual.merger.ln_out": 1,
            "visual.merger.q_proj": 1,
            "visual.merger.k_proj": 1,
            "visual.merger.v_proj": 1,
            "visual.merger.out_proj": 1,
        }
        
        # Distribute visual blocks between GPU 0 and 1
        for i in range(32):
            if i < 16:  # First half of visual blocks to GPU 0
                device_map[f"visual.blocks.{i}"] = 0
                device_map[f"visual.norm.{i}"] = 0
            else:  # Second half to GPU 1
                device_map[f"visual.blocks.{i}"] = 1
                device_map[f"visual.norm.{i}"] = 1
        
        # Distribute language model layers
        layers_gpu0 = total_layers // 3  # Fewer layers on GPU 0 due to visual components
        layers_gpu1 = (total_layers - layers_gpu0) // 2
        layers_gpu2 = total_layers - layers_gpu0 - layers_gpu1
        
        current_layer = 0
        
        # GPU 0 layers
        for i in range(layers_gpu0):
            device_map[f"model.layers.{current_layer}"] = 0
            device_map[f"model.layers.{current_layer}.self_attn.rotary_emb"] = 0
            current_layer += 1
        
        # GPU 1 layers
        for i in range(layers_gpu1):
            device_map[f"model.layers.{current_layer}"] = 1
            device_map[f"model.layers.{current_layer}.self_attn.rotary_emb"] = 1
            current_layer += 1
        
        # GPU 2 layers
        for i in range(layers_gpu2):
            device_map[f"model.layers.{current_layer}"] = 2
            device_map[f"model.layers.{current_layer}.self_attn.rotary_emb"] = 2
            current_layer += 1
        
        # Final layers on last GPU
        device_map.update({
            "model.norm": 2,
            "lm_head": 2
        })
    
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Configure PyTorch's CUDA allocator
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
    
    print("\nDevice Map:")
    for key, value in device_map.items():
        print(f"{key}: GPU {value}")
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        **model_kwargs
    )
    
    return model

processor = AutoProcessor.from_pretrained(model_path)

def process_input(image, video, prompt, temperature=0.8, top_k=50, top_p=0.9, max_tokens=100):
    if image is not None:
        media_type = "image"
        media = image
    elif video is not None:
        media_type = "video"
        media = video
    else:
        return "Please upload an image or video."
    
    # Clear cache before processing
    torch.cuda.empty_cache()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": media_type, media_type: media},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    device = next(model.parameters()).device
    
    with torch.cuda.amp.autocast():  # Use automatic mixed precision
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device and convert to float16
        inputs = {k: v.to(device, dtype=torch.float16 if v.dtype == torch.float32 else v.dtype) 
                 for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_cache=True  # Enable KV cache
        )
    
    # Clear cache after processing
    torch.cuda.empty_cache()
    
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    response = output_text[0].split("assistant\n")[-1].strip()
    return response

def create_interface():
    interface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Image(type="filepath", label="Upload Image (optional)"),
            gr.Video(label="Upload Video (optional)"),
            gr.Textbox(label="Text Prompt"),
            gr.Slider(0.1, 1.0, value=0.8, label="Temperature"),
            gr.Slider(1, 100, value=50, step=1, label="Top-k"),
            gr.Slider(0.1, 1.0, value=0.9, label="Top-p"),
            gr.Slider(1, 500, value=100, step=10, label="Max Tokens")
        ],
        outputs=gr.Textbox(label="Generated Description"),
        title="Qwen2-VL-72B Vision-Language Model",
        description="Upload an image or video and enter a prompt to generate a description.",
    )
    return interface

def print_gpu_info():
    num_gpus = torch.cuda.device_count()
    print(f"\nNumber of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2-VL model with optional Flash Attention 2")
    parser.add_argument("--flash-attn2", action="store_true", help="Use Flash Attention 2")
    parser.add_argument("--memory-fraction", type=float, default=0.95, help="Maximum CUDA memory fraction to use")
    args = parser.parse_args()
    
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    print_gpu_info()
    model = load_model(use_flash_attention=args.flash_attn2)
    interface = create_interface()
    interface.launch(share=True)
