```
# Install dependencies
sudo apt update
sudo apt install build-essential zlib1g-dev libncurses5-dev \
  libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
  libffi-dev libsqlite3-dev wget libbz2-dev -y

sudo apt install -y liblzma-dev lzma

# Download Python 3.9 source
cd /tmp
wget https://www.python.org/ftp/python/3.9.18/Python-3.9.18.tar.xz
tar -xf Python-3.9.18.tar.xz
cd Python-3.9.18

# Configure and build 
./configure --enable-optimizations --enable-shared --with-lto --disable-test-modules --with-system-libmpdec
make -j$(nproc)
sudo make altinstall

# Verify
python3.9 --version

# Add to library path
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/python3.9.conf
sudo ldconfig


python3.9 -m venv venv_3.9

pip install --upgrade pip

pip install ultralytics --no-deps

pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu117
pip install onnx coremltools onnxruntime onnxsim

ONNX export success, saved as model_compatible_full.onnx
Minimum required torch version for importing coremltools.optimize.torch is 2.1.0. Got torch version 1.13.1.
Failed to load _MLModelProxy: No module named 'coremltools.libcoremlpython'

```

```

sudo apt update
sudo apt install build-essential libssl-dev libffi-dev

# First, install SSL development libraries
sudo apt-get update
sudo apt-get install libssl-dev libffi-dev

# If you installed Python from source, recompile it
wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tgz
tar -xvf Python-3.10.13.tgz
cd Python-3.10.13
cd /tmp/Python-3.10.13
sudo make clean
./configure --enable-optimizations --with-ssl-default-suites=openssl
make -j $(nproc)
sudo make altinstall

python3.10 -m venv venv_3.10

pip install --upgrade pip
pip install --upgrade setuptools pip wheel

python3.10 -m venv venv --upgrade-deps

(venv) root@97188a432527:/tmp# python -c "import pkg_resources; print('Fixed')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'pkg_resources'
```