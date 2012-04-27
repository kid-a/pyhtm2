PYTHON = python2.7
MAIN = main.py
TEST = py.test

run:
	$(PYTHON) $(MAIN)

test:
	$(TEST)

clean:
	-rm *.pyc *.bmp

clearbak:
	-rm *~
