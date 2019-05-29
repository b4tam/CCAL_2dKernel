PY := python

.PHONY: information_coefficient test

information_coefficient: information_coefficient.pyx 
	$(PY) setup.py build_ext --inplace

test: information_coefficient kernel2d.py
	$(PY) kernel2d.py