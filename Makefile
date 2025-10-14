# Makefile for LaTeX project

# Compiler
LATEXMK = latexmk -pdf

# Main file
MAIN = main.tex

# Default target
all: $(MAIN)
	$(LATEXMK) $(MAIN)

# Clean up auxiliary files
clean:
	$(LATEXMK) -c
	rm -f *.pdf

.PHONY: all clean
