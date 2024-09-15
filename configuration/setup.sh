# Create and activate virtual environment
if [ ! -d "venv" ]; then
  YAML_FILE="configuration/python_path.yaml"
  PYTHON_PATH=$(grep 'python_path' "$YAML_FILE" | sed 's/.*: \(.*\)/\1/')
  VENV_DIR="venv"
  $PYTHON_PATH -m venv "$VENV_DIR"
  echo "Virtual environment 'venv' created."
else
  :
fi

. venv/Scripts/activate
echo "Virtual environment correctly activated"


# Install requirements
pip install -r requirements.txt
cd venv/Lib/site-packages
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e . -q
cd ../../../..
mkdir checkpoints
curl -o checkpoints/sam2_hiera_tiny.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "Requirements correctly installed"