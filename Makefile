PY := python

information_coefficient: information_coefficient.pyx 
	$(PY) setup.py build_ext --inplace
