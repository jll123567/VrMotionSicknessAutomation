PS2PDF_OPTS=-dEmbedAllFonts=true -dPDFSETTINGS=/prepress

all: thesis.pdf

datestamp:
	rm -f ../THIS-IS-VERSION*
	touch ../THIS-IS-VERSION-`date +%F-%T`

view: thesis.ps
	gv thesis.ps

thesis.dvi: *.bib Figures/*.eps *.tex *.cls
	( latex thesis || rm -f thesis.dvi ) && \
	makeglossaries thesis && \
	bibtex thesis && \
	( latex thesis || rm -f thesis.dvi ) && \
	( latex thesis || rm -f thesis.dvi ) && \
	( latex thesis || rm -f thesis.dvi )

thesis.ps: thesis.dvi
	dvips -t letter -o thesis.ps thesis

thesis.pdf: thesis.ps
	ps2pdf ${PS2PDF_OPTS} thesis.ps thesis.pdf

clean:
	rm -vf *.aux *.bbl *.blg *.dvi *.ps *.pdf *.log *.lot *.toc \
	*.lof *.gls *.glg *.ist *.glo *~
