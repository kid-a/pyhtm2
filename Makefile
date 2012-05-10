PYTHON = python
MAIN = main.py
TEST = py.test
CHARTS = charts

run:
	$(PYTHON) $(MAIN)

test:
	$(TEST)

charts:
	mv activation-sigma* $(CHARTS)
	cd $(CHARTS)
	make

clean:
	-rm *.pyc *.bmp

clearbak:
	-rm *~
