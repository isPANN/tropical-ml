# Makefile for tropical-pruning
# Manages dependencies, building, and experiments

TROPICAL_GEMM_REPO := https://github.com/TensorBFS/tropical-gemm.git
TROPICAL_GEMM_DIR := ../tropical-gemm
PYTHON := uv run python
MATURIN := maturin

.PHONY: help setup setup-cpu setup-cuda deps deps-cpu deps-cuda deps-cuda-direct \
        test run clean clean-deps check-cuda verify

#==============================================================================
# Help
#==============================================================================

help:
	@echo "tropical-pruning Makefile"
	@echo ""
	@echo "Setup (first time):"
	@echo "  make setup           - Setup with CPU-only tropical-gemm"
	@echo "  make setup-cuda      - Setup with CUDA-enabled tropical-gemm"
	@echo ""
	@echo "Dependencies:"
	@echo "  make deps            - Install/rebuild tropical-gemm (CPU)"
	@echo "  make deps-cuda       - Install/rebuild tropical-gemm (CUDA)"
	@echo "  make clone-gemm      - Clone tropical-gemm repo only"
	@echo ""
	@echo "Development:"
	@echo "  make test            - Run tests"
	@echo "  make run             - Run MNIST experiment"
	@echo "  make verify          - Verify tropical-gemm installation"
	@echo "  make check-cuda      - Check CUDA availability"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean           - Clean build artifacts"
	@echo "  make clean-deps      - Remove tropical-gemm clone"

#==============================================================================
# Setup (first time)
#==============================================================================

setup: clone-gemm
	uv sync --dev
	$(MAKE) deps
	@echo ""
	@echo "Setup complete (CPU)! Run 'make run' to test."

setup-cuda: clone-gemm
	uv sync --dev
	$(MAKE) deps-cuda
	uv sync
	@echo ""
	@echo "Setup complete (CUDA)! Run 'make check-cuda' to verify."

#==============================================================================
# Dependencies
#==============================================================================

clone-gemm:
	@if [ ! -d "$(TROPICAL_GEMM_DIR)" ]; then \
		echo "Cloning tropical-gemm..."; \
		git clone $(TROPICAL_GEMM_REPO) $(TROPICAL_GEMM_DIR); \
	else \
		echo "tropical-gemm already exists at $(TROPICAL_GEMM_DIR)"; \
	fi

# Build and install tropical-gemm (CPU only)
deps: clone-gemm
	@echo "Building tropical-gemm (CPU)..."
	cd $(TROPICAL_GEMM_DIR)/crates/tropical-gemm-python && \
		$(MATURIN) build --release && \
		$(PYTHON) -m pip install target/wheels/*.whl --force-reinstall
	@echo "tropical-gemm (CPU) installed."

# Build and install tropical-gemm with CUDA
deps-cuda: clone-gemm
	@echo "Building tropical-gemm (CUDA)..."
	@echo "Note: If this hangs, try: make deps-cuda-direct"
	@which maturin >/dev/null 2>&1 || (echo "Installing maturin..." && pip install maturin)
	cd $(TROPICAL_GEMM_DIR)/crates/tropical-gemm-python && \
		maturin build --release --features cuda && \
		$(PYTHON) -m pip install target/wheels/*.whl --force-reinstall
	@echo "tropical-gemm (CUDA) installed."

# Alternative: Build directly with maturin and system python (bypasses uv)
deps-cuda-direct: clone-gemm
	@echo "Building tropical-gemm (CUDA) - direct method (bypasses uv)..."
	@which maturin >/dev/null 2>&1 || (echo "Installing maturin..." && pip install maturin)
	@which python3 >/dev/null 2>&1 || (echo "Error: python3 not found" && exit 1)
	cd $(TROPICAL_GEMM_DIR)/crates/tropical-gemm-python && \
		maturin build --release --features cuda && \
		python3 -m pip install target/wheels/*.whl --force-reinstall
	@echo "tropical-gemm (CUDA) installed."

# Update tropical-gemm to latest
update-gemm:
	@if [ -d "$(TROPICAL_GEMM_DIR)" ]; then \
		echo "Updating tropical-gemm..."; \
		cd $(TROPICAL_GEMM_DIR) && git pull; \
	else \
		echo "tropical-gemm not found. Run 'make clone-gemm' first."; \
	fi

#==============================================================================
# Development
#==============================================================================

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(PYTHON) examples/mnist_tropical_pruning.py --skip-training

run-train:
	$(PYTHON) examples/mnist_tropical_pruning.py --epochs 10

verify:
	@$(PYTHON) -c "\
import tropical_gemm as tg; \
import numpy as np; \
print('Available functions:', [x for x in dir(tg) if not x.startswith('_')]); \
a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32); \
b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32); \
c = tg.maxplus_matmul(a, b); \
print('MaxPlus matmul test: PASSED'); \
c_arg, arg = tg.maxplus_matmul_with_argmax(a, b); \
print('MaxPlus with argmax: PASSED')"

check-cuda:
	@$(PYTHON) -c "import tropical_gemm as tg; print(f'CUDA available: {tg.cuda_available()}') if hasattr(tg, 'cuda_available') else print('CUDA support not compiled (cuda_available not found)')"

#==============================================================================
# Cleanup
#==============================================================================

clean:
	rm -rf __pycache__ .pytest_cache
	rm -rf tropical_pruning/__pycache__
	rm -rf tests/__pycache__
	rm -rf *.egg-info dist build
	rm -f mnist_mlp.pt
	rm -rf data/

clean-deps:
	rm -rf $(TROPICAL_GEMM_DIR)
	@echo "Removed tropical-gemm. Run 'make clone-gemm' to re-clone."
