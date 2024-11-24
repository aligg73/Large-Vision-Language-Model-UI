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
        # More aggressive layer distribution across GPUs
        layers_per_gpu = total_layers // num_gpus
        remaining_layers = total_layers % num_gpus
        
        device_map = {
            # Vision components on GPU 0
            "visual": 0,
            "vision_model": 0,
            "vision_projection": 0,
            "model.embed_tokens": 0,
            "perceiver": 0,
            "image_processor": 0,
            "rotary_emb": 0,
            "model.rotary_emb": 0,
        }
        
        # Distribute layers more evenly
        current_layer = 0
        for gpu_id in range(num_gpus):
            # Add extra layer to early GPUs if division wasn't even
            extra_layer = 1 if gpu_id < remaining_layers else 0
            gpu_layers = layers_per_gpu + extra_layer
            
            # Assign this GPU's layers
            for i in range(current_layer, current_layer + gpu_layers):
                device_map[f"model.layers.{i}"] = gpu_id
                device_map[f"model.layers.{i}.self_attn.rotary_emb"] = gpu_id
            
            current_layer += gpu_layers
        
        # Final layers on last GPU
        device_map.update({
            "model.norm": num_gpus - 1,
            "lm_head": num_gpus - 1
        })
        
        # Visual blocks on first GPU
        for i in range(32):
            device_map[f"visual.blocks.{i}"] = 0
            device_map[f"visual.norm.{i}"] = 0
    
    model_kwargs = {
        "torch_dtype": torch.float16,  # Use FP16
        "device_map": device_map,
        "low_cpu_mem_usage": True,     # Reduce CPU memory usage
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    # Set PyTorch memory allocator settings
    torch.cuda.set_per_process_memory_fraction(0.95)  # Leave some headroom
    torch.cuda.empty_cache()  # Clear cache before loading
    
    # Configure PyTorch's CUDA allocator
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
    
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
