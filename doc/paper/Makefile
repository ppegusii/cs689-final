paper = cs689_report
PDF = $(paper).pdf
TEX = $(paper).tex evaluation.tex relatedwork.tex

pdf:	$(TEX) 
	pdflatex $(paper)
	pdflatex $(paper)
	bibtex $(paper)
	pdflatex $(paper)
	pdflatex $(paper)

clean:
	rm -f $(PS) $(PDF) *.dvi *.log *.aux *.bbl *.blg
