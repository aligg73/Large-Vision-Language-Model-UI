echo "is CUDA 12.4 being used?"
python -V
nvidia-smi
# git clone https://github.com/Kaszebe/Large-Vision-Language-Model-UI.git
# cd Large-Vision-Language-Model-UI
# python3 -m venv qwen_venv
source qwen_venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install transformers accelerate qwen-vl-utils gradio autoawq
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
export QWEN_MODEL_PATH="/workspace/Large-Vision-Language-Model-UI/model72b"
python run_qwen_model.py --flash-attn2