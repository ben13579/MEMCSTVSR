# MEMCSTVSR
## Setup

### Prerequisites
/b python 3.11.13

### Installation
```
git clone https://github.com/ben13579/MEMCSTVSR.git
cd MEMCSTVSR
conda env create -f requirements.yml
pip install -e .
```
install flash attention
```
pip install flash-attn==2.8.3 \
  --no-build-isolation \
  --no-cache-dir
```

## Usage
This project is based on diffsynth(幫我變成內嵌入連結https://github.com/modelscope/diffsynth-studio) 
基本的設定以及實作可參考README_wan.md
### DiT training and validation
### DiT inference
### IND training
