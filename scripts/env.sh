# Source before running anything: python env + GPU Vulkan for SAPIEN rendering.
# Vulkan fix from the AutoDL docs (best practices -> Vulkan): headless NVIDIA
# Vulkan needs the EGL library as ICD; see /etc/vulkan/icd.d/my_nvidia_icd.json.
source /root/autodl-tmp/paf_venv/bin/activate
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/my_nvidia_icd.json
export PYTHONPATH=/root/autodl-tmp/port_align_field:$PYTHONPATH
export PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /root/autodl-tmp/port_align_field
