import os
import sys

# -- Project information -----------------------------------------------------
project = 'T3Toolbox'
copyright = '2026, Nick Alger and Blake Christierson'
author = 'Nick Alger and Blake Christierson'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    #'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'autoapi.extension',
    'sphinx.ext.githubpages',
]

# 1. HIDE MODULE NAMES IN SIGNATURES (CRITICAL)
# This prevents Sphinx from prefixing every class/function with its module path.
add_module_names = False
typehints_fully_qualified = False
python_use_unqualified_type_names = True
toc_object_entries_show_parents = 'hide'

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ['../t3toolbox']  # Path to your source code
autoapi_template_dir = '_templates/autoapi'
autoapi_type = 'python'

# 2. PREVENT FULL NAMES IN SIDEBAR/TOC
# By default, AutoAPI might include the full path in titles.
autoapi_add_toctree_entry = True
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]

autoapi_own_page_level = 'method'

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'

