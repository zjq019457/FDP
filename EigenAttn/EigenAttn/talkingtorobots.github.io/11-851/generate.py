import os
import yaml
import openpyxl
import argparse
from copy import copy
from jinja2 import Template
from openpyxl.styles import Border, Side
from openpyxl.utils import range_boundaries, get_column_letter
from openpyxl.worksheet.table import Table


papers = yaml.load(open("papers.yaml"), Loader=yaml.CLoader)
with open("papers.jinja2", 'r') as file:
  template = Template(file.read())

  rendered = template.render(papers=papers)
  with open("papers.html", 'wt') as output_file:
      output_file.write(rendered)
