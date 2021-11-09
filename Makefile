install:
	@echo "Checking to make sure submodules are cloned..."
	git submodule update --init --recursive
	@echo "Installing local python libraries..."
	pip install lib/*
	@echo "Installing clustering hyperparameters..."
	pip install -e .
	@echo "Done!"
