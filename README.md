# FinTech 545 --- Quantitative Risk Management

Lecture notes, code, and problem sets for FinTech 545. Nine weeks of notes are
written in Quarto and rendered to PDF, and the same nine sources are assembled
into a single book.

## Layout

| Path | What is there |
|:--|:--|
| `Week01` ... `Week09` | One directory per lecture. The `.qmd` source, its rendered PDF, `figures/`, the Julia code for that week, and any data it uses. |
| `book/` | The nine weeks compiled into one PDF. Generated, not written by hand. |
| `library/` | Shared Julia code. Weeks that use a file generally carry their own copy. |
| `Projects/` | The problem sets. |
| `MidTerm/`, `Final/`, `QuizFiles/`, `OldFinals/`, `OldExtraCredit/` | Assessments. |
| `notes-format-spec.md` | The format every week's `.qmd` follows. Read it before editing a week. |

## Building

From the repository root:

| Command | What it does |
|:--|:--|
| `make` | Render every week to PDF. |
| `make book` | Regenerate `book/book.qmd` from the nine weeks and render it. |
| `make html` | Render every week to HTML. |
| `make slides` | Render every reveal.js deck to HTML. |
| `make figures` | Rebuild the matplotlib and TikZ figures. |
| `make clean` | Remove rendered output. |

`make` and `make book` are the two you will use. One week on its own:

```
cd Week05 && quarto render week05.qmd --to pdf
```

Each week's YAML names its own output file, so the PDF lands beside the source
as `Week 05 - Expected Shortfall and Copulas.pdf` rather than `week05.pdf`.

### Slides

Decks live in `Week0N/slides/`, one level below the notes. That keeps them out of
the `Week*/week*.qmd` wildcard `make` uses, which would otherwise try to render a
deck as an article PDF, and out of the way of `make clean`. Weeks 02 through 05
have decks so far. One on its own:

```
quarto render Week05/slides/week05-slides.qmd --to revealjs
```

A deck reads its figures from `../figures/`, so nothing is duplicated.

## What you need installed

**Quarto.** The build is `quarto render` for every target. Install from
[quarto.org](https://quarto.org/docs/get-started/).

**A TeX distribution with lualatex.** The notes set `pdf-engine: lualatex` and
`documentclass: scrartcl`, and load `float`, `needspace`, and `etoolbox`. On
Debian or Ubuntu:

```
apt-get install texlive-latex-recommended texlive-latex-extra \
                texlive-luatex lmodern fonts-lmodern \
                librsvg2-bin poppler-utils
```

`texlive-luatex` is the one that gets missed. Without it lualatex reports that
it cannot load `lmroman10-regular` or cannot find `luaotfload-main`. Both read
like a missing font and neither is. `librsvg2-bin` provides `rsvg-convert`,
which Quarto uses to place the SVG figures. `poppler-utils` provides
`pdftocairo` and `pdftotext`, covered below.

That line is Debian and Ubuntu. On macOS the equivalent is MacTeX plus
`brew install librsvg poppler`, and on Windows it is MiKTeX or TeX Live, though
building under WSL avoids having to work that out.

**Python 3**, for the figure scripts and for `book/make_book.py`. `make_book.py`
uses only the standard library. The figure scripts need what is in
`requirements.txt`:

```
pip install -r requirements.txt
```

Rendered figures are committed, so a fresh clone builds the notes without ever
running Python. You only need it when a figure changes.

**pdflatex and pdftocairo**, for the TikZ figures. Neither is a Python package.
`pdflatex` is part of the TeX distribution above, from `texlive-binaries`, and
`pdftocairo` comes from `poppler-utils` in the same apt line, so both are already
installed by that point. `poppler-utils` also supplies `pdftotext`, which the
cross-reference check further down uses. The generated SVGs are committed, so
this pair only matters when a diagram changes.

**Julia**, for the course code. It is not part of the notes build. Most weeks
that carry code also carry a `Project.toml`:

```
cd Week05 && julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Figures

Two kinds, both committed.

matplotlib PNGs, built by `figures/make_figures.py` in Weeks 02 through 07 and
Week 09. Each script writes only its own week's figures and prints the numbers
it plotted, so a figure or a table in the notes can be checked against what the
script reports.

TikZ diagrams, in Weeks 06, 07, and 09, built from `.tex` sources through
`pdflatex` and `pdftocairo` into SVG.

`make figures` does both for every week.

## The book

`book/make_book.py` reads the nine `Week0N/week0N.qmd` files and writes
`book/book.qmd`. The weekly files are the single source of truth and the script
never modifies them, so editing a week and rebuilding updates the week's PDF and
the book together. Do not edit `book/book.qmd` -- it is overwritten on every
build. `book/README.md` covers the transform and the checks it runs.

## Editing a week

`notes-format-spec.md` defines the format: the YAML header, the heading
hierarchy, captioned tables and figures, and the cross-reference rules.

The rule worth repeating here is that a section number is never written as
literal text. Sections get renumbered whenever one is added or moved, and a hard
coded "Section 5.4" goes wrong silently. Label the heading and reference the
label:

```
## Student's t Distribution {#sec-t}

... @sec-t covers the distribution where this matters most for us.
```

After editing, render the week and confirm that nothing came out as `??`. That
is what an unresolved cross-reference looks like in the PDF, and Quarto does not
fail the build over it:

```
quarto render week05.qmd --to pdf
pdftotext "Week 05 - Expected Shortfall and Copulas.pdf" - | grep -c '??'
```

## Conventions

**255 trading days per year**, not 252. It runs through the notes, the code, and
the problem sets.
