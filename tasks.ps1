# Usage: .\tasks.ps1 install | test | notebooks | pdf
param(
    [Parameter(Position = 0)]
    [string]$Task = "help"
)

$Root = $PSScriptRoot
Set-Location $Root

switch ($Task) {
    "install" {
        py -3 -m pip install -r requirements.txt
    }
    "test" {
        py -3 -m pytest tests -v
    }
    "notebooks" {
        Set-Location notebooks
        jupyter nbconvert --execute --inplace 01_EDA.ipynb 02_Modeling.ipynb
        Set-Location $Root
    }
    "pdf" {
        py -3 scripts/build_presentation_pdf.py
    }
    "emit02" {
        py -3 scripts/emit_02_notebook.py
    }
    "clean" {
        Get-ChildItem -Path $Root -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Caches Python (__pycache__) supprimes."
    }
    default {
        Write-Host "Taches: install, test, notebooks, pdf, emit02, clean"
    }
}
