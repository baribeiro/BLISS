.PHONY: install install-torch install-sfno install-physicsnemo install-rest test lint build clean

TORCH_INDEX = https://download.pytorch.org/whl/cu128

install: install-torch install-sfno install-physicsnemo install-rest  ## full environment, in the order that works

install-torch:          ## PyTorch 2.9.1 with CUDA 12.8
	pip install torch==2.9.1 torchvision==0.24.1 --index-url $(TORCH_INDEX)

install-sfno:           ## torch-harmonics 0.8.0 (SFNO), compiled against the installed torch; needs a C++ compiler and nvcc
	pip install --no-build-isolation --no-deps --no-binary torch-harmonics torch-harmonics==0.8.0

install-physicsnemo:    ## NVIDIA PhysicsNeMo 2.2.2 (FigConvNet); its metadata asks for torch>=2.10, it runs on 2.9.1
	pip install --no-deps nvidia-physicsnemo==2.2.2
	pip install -r requirements-physicsnemo.txt -c constraints.txt
	pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

install-rest:           ## remaining dependencies and the bliss package (loader and scorer)
	pip install -r requirements.txt -c constraints.txt
	pip install -e . -c constraints.txt

test:                   ## run the test suite
	python -m pytest -q tests

lint:                   ## check the package with ruff
	ruff check bliss tests

build:                  ## build the source and wheel distributions
	python -m build

clean:
	rm -rf build dist *.egg-info .pytest_cache
