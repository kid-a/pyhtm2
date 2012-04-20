PYTHON = python
MAIN = main.py
TEST = py.test

# run:
# 	$(PYTHON) $(MAIN)

test:
	$(TEST)

clean:
	-rm *.pyc

clearbak:
	-rm *~
