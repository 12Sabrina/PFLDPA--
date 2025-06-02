# PFLDPA--
1. **Create and activate a conda environment**:
   ```bash
   conda create -n FedPy39 python=3.9
   conda activate FedPy39
   conda init
   source ~/.bashrc
   conda activate FedPy39
2. **Install the core dependencies:**
    ```bash
    pip install tensorflow==2.15.0 tensorflow-estimator==2.15.0 tensorflow-privacy==0.8.11 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install tensorflow-probability==0.23.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
    pip install fedlab torch
