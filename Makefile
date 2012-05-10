PYTHON = python
MAIN = main.py
TEST = py.test
UTILS = utils.py

run:
	$(PYTHON) $(MAIN)

test:
	$(TEST)

profile-utils:
	$(PYTHON) $(UTILS)

clean:
	-rm *.pyc *.bmp

clearbak:
	-rm *~
