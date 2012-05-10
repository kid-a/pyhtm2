PYTHON = python
MAIN = main.py
TEST = py.test
UTILS = utils.py
SC = spatial_clustering.py

run:
	$(PYTHON) $(MAIN)

test:
	$(TEST)

profile-utils:
	$(PYTHON) $(UTILS)

profile-spatial-clustering:
	$(PYTHON) $(SC)

clean:
	-rm *.pyc *.bmp

clearbak:
	-rm *~
