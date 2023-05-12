remove_unused_imports:
	autoflake . --recursive --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports

sort_imports:
	isort . --atomic

format_pyfiles:
	black .

format_package: remove_unused_imports sort_imports format_pyfiles

