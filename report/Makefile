
RMD=$(wildcard *.Rmd)
IPYNB=$(wildcard *.ipynb)
RMD_HTML=$(patsubst %.Rmd,%.html,$(RMD))
IPYNB_HTML=$(patsubst %.ipynb,%.html,$(IPYNB))
DATADIR=../data
_CLUSDATA=blobs.csv noisy_moons.csv noisy_circles.csv
CLUSDATA=$(patsubst %,$(DATADIR)/%,$(_CLUSDATA))

all: data $(RMD_HTML) $(IPYNB_HTML)

data: $(CLUSDATA)

site: data $(IPYNB_HTML)
	R -e "rmarkdown::render_site()"

$(CLUSDATA):
	python gen-cluster-data.py

%.html: %.Rmd
	R -e "rmarkdown::render('$^')"

%.html: %.ipynb
	jupyter nbconvert --to html $^

clean:
	rm -f $(RMD_HTML) $(IPYNB_HTML) $(CLUSDATA)
	R -e "rmarkdown::clean_site()"


