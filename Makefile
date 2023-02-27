initialize_pipenv_venv:
	pipenv --python 3.9

install_cpu:
	cp cpu-Pipfile Pipfile\
	&& pipenv clean\
	&& pipenv install\
	&& pipenv run pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

install_gpu:
	cp gpu-Pipfile Pipfile\
	&& pipenv clean\
	&& pipenv install\
	&& pipenv run pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
