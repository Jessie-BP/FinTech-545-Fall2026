# Build the FinTech 545 lecture notes.
#   make          -> all PDFs (named from each .qmd's output-file)
#   make book     -> the nine weeks assembled into one PDF, in book/
#   make html     -> HTML versions
#   make slides   -> the reveal.js decks
#   make figures  -> recompile the TikZ figures to SVG and the data-driven PNGs
#   make clean    -> remove rendered output
QMD := $(wildcard Week*/week*.qmd)

all:
	for f in $(QMD); do quarto render $$f --to pdf; done

# The single-volume book. make_book.py regenerates book/book.qmd from the nine
# weekly .qmd files, so the book always matches the weeks.
book:
	cd book && python3 make_book.py && quarto render book.qmd --to pdf

html:
	for f in $(QMD); do quarto render $$f --to html; done

# Slide decks live one level deeper than the notes so the QMD wildcard above
# does not pick them up and try to render them as articles.
SLIDES := $(wildcard Week*/slides/week*-slides.qmd)

slides:
	for f in $(SLIDES); do quarto render $$f --to revealjs; done

figures:
	cd Week02/figures && python3 make_figures.py
	cd Week03/figures && python3 make_figures.py
	cd Week04/figures && python3 make_figures.py
	cd Week05/figures && python3 make_figures.py
	cd Week06/figures && for f in tree1 tree2;   do pdflatex -interaction=nonstopmode $$f.tex && pdftocairo -svg $$f.pdf $$f.svg; done
	cd Week06/figures && python3 make_figures.py
	cd Week07/figures && for f in amput divtree; do pdflatex -interaction=nonstopmode $$f.tex && pdftocairo -svg $$f.pdf $$f.svg; done
	cd Week07/figures && python3 make_figures.py
	cd Week09/figures && for f in taxonomy sigma simflow; do pdflatex -interaction=nonstopmode $$f.tex && pdftocairo -svg $$f.pdf $$f.svg; done
	cd Week09/figures && python3 make_figures.py

clean:
	rm -f book/book.qmd book/*.pdf book/*.html
	rm -rf book/*_files
	rm -f Week*/*.pdf Week*/*.html
	rm -rf Week*/*_files Week*/figures/*.aux Week*/figures/*.log Week*/figures/*.pdf
	rm -rf Week*/slides/*.html Week*/slides/*_files

.PHONY: all book html slides figures clean
