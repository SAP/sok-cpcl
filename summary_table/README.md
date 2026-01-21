# Summary Table for SoK: Enhancing Cryptographic Collaborative Learning with Differential Privacy

This folder contains the LaTeX source code to generate the summary table presented in our SoK paper "Enhancing Cryptographic Collaborative Learning with Differential Privacy". The table categorizes and compares different approaches for encrypted and differentially private collaborative learning.

## Structure
- `main.tex`: Main LaTeX file to compile the summary table.
- `macros.tex`: LaTeX macros used in the summary table.
- `bibliography.bib`: Bibliography file containing references cited in the summary table.
- `table.png`: Rendered image of the summary table.
- `main.pdf`: Compiled PDF of the summary table.

## Requirements and Setup

### LaTeX Installation

You need a complete LaTeX distribution with standard packages. Choose one based on your platform:

**macOS:**
```bash
brew install mactex
# or: brew install mactex-no-gui (lightweight version without GUI)
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install texlive-full
# or: sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-latex-extra texlive-bibtex-extra
```

### Build the Summary Table

Navigate to this folder and compile with:

```bash
cd summary_table
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

This generates `main.pdf` with the compiled summary table. The `bibtex` step processes references from `bibliography.bib`.

**Or use a single command:**
```bash
latexmk -pdf main.tex
```
(Requires `latexmk`, which is included in most LaTeX distributions)

### Generated Outputs

- `main.pdf` — Compiled PDF of the summary table

## Contributing
Contributions to enhance and update the summary table are welcome. Please ensure that the LaTeX compiles correctly, and the table remains well-formatted. Follow the contribution guidelines provided in `CONTRIBUTING.md`.
