
RMD=$(wildcard *.Rmd)
IPYNB=$(wildcard *.ipynb)
RMD_HTML=$(patsubst %.Rmd,%.html,$(RMD))
IPYNB_HTML=$(patsubst %.ipynb,%.html,$(IPYNB))

all: $(RMD_HTML) $(IPYNB_HTML)

%.html: %.Rmd
	R -e "rmarkdown::render('$^')"

%.html: %.ipynb
	jupyter nbconvert --to html $^

clean:
	rm -f $(RMD_HTML) $(IPYNB_HTML)
