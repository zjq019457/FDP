import os
import yaml
import openpyxl
import argparse
from copy import copy
from jinja2 import Template
from openpyxl.styles import Border, Side
from openpyxl.utils import range_boundaries, get_column_letter
from openpyxl.worksheet.table import Table

parser = argparse.ArgumentParser(description='Generate Website, CV, COA')
parser.add_argument('--onepager', action='store_true',
                    help='One-pager vs full')
args = parser.parse_args()


# Load Publications 
if args.onepager:
    pubs = yaml.load(open("yaml/publications_1p.yaml"), Loader=yaml.CLoader)
else:
    pubs = yaml.load(open("yaml/publications.yaml"), Loader=yaml.CLoader)

# Load Author urls
webs = yaml.load(open("yaml/websites.yaml"), Loader=yaml.CLoader)

# Students
student_yaml = yaml.load(open("yaml/students/phd.yaml"), Loader=yaml.CLoader)
student_names = set([s["NAME"] for s in student_yaml])

# Theses
theses_yaml = yaml.load(open("yaml/theses.yaml"), Loader=yaml.CLoader)
teach_yaml = yaml.load(open("yaml/teaching.yaml"), Loader=yaml.CLoader)
areachair_yaml = yaml.load(open("yaml/area_chair.yaml"), Loader=yaml.CLoader)

# Load template
types = {
         "Conference": ["inproceedings", "booktitle"],
         "Journal": ["article", "journal"],
         "Preprint": ["article", "journal"],
         "Workshop": ["inproceedings", "booktitle"],
        }
colors = {
          "WS1":"primary", 
          "WS2":"primary", 
          "WS3": "success", 
          "WS4": "danger", 
          "WS5": "warning", 
          "O": "info", 
          "A": "secondary"
        }

accents = {"{\\'e}": "é", "{\\`e}": "è", "{\\'a}": "á", "{\\'o}": "ó", 
           "{\\'u}": "ú", "{\\'c}": "ć", "{\\'\\\\i}":"í",
           "{\\\"a}": "ä", "{\\o}": "ø", "{\\aa}": "å", "{\\l}": "ł", "{\\'y}": "ý",
           "{\\\"o}": "ö","{\\\"u}": "ü", "{\\'s}": "ś", "{\\^o}": "ô",
           "\\v{c}": "č", "\\v{s}": "š", "\\v{r}": "ř"}

def pretty_author_names(s):
  for k in accents:
    s = s.replace(k, accents[k])
    s = s.replace(k.upper(), accents[k].upper())
  if s[0] == "{" and s[-1] == "}":
    s = s[1:-1]
  return s

def update_pub_entry(entry):
  # Authors and links to all co-authors
  entry["AUTHORS-PRETTY"] = pretty_author_names(", ".join(entry["AUTHORS"]))
  for v in webs:
    entry["AUTHORS-PRETTY"] = entry["AUTHORS-PRETTY"].replace(v, "<a href={}>{}</a>".format(webs[v], v))
  for p in student_names:
    entry["AUTHORS-PRETTY"] = entry["AUTHORS-PRETTY"].replace(p, f"<div class=\"name\">{p}</div>")
  entry["AUTHORS-PRETTY"] = entry["AUTHORS-PRETTY"].replace("Yonatan Bisk", f"<div class=\"name\">Yonatan Bisk</div>")

def generate_shorts():
    for pub in pubs:
        if "CUTE" in pub:
            fname = f"../r/{pub['CUTE']}.html"
            if os.path.exists(fname):
                os.remove(fname)
            o = open(fname,'wt')
            o.write('<meta http-equiv="Refresh" content="0; url=\'{pub["URL"]}\'" />')
            o.close()

def generate_publications_website():
    with open("templates/pub_template.jinja2", 'r') as file:
      pub_template = Template(file.read())

    for entry in pubs:
      update_pub_entry(entry)

    ## Generate Plot ##
    data = {"WS1":0, "WS2":0, "WS3":0, "WS4":0, "WS5":0, "O":0}
    for entry in pubs:
        u = 1.0
        data[entry["FIELD"]] += u

    rendered = pub_template.render(publications=pubs, data=data, colors=colors, types=types)
    with open("../publications.html", 'wt') as output_file:
        output_file.write(rendered)

def number_pub(vals, pub_count=1):  
    for val in vals:
        val["idx"] = pub_count
        pub_count += 1
    return pub_count
        
def generate_CV():
    ## Generate CV ##
    if args.onepager:
        template = "templates/CV_template_1p.jinja2"
    else:
        template = "templates/CV_template.jinja2"
    with open(template, 'r') as file:
      latex_template = Template(file.read())
    latex_papers = {"Journal": [],
                    "Conference": [],
                    "Workshop": [],
                    "Preprint": [],
                    }
    for entry in pubs:
      latex_papers[entry["TYPE"]].append(entry)

    # Hi dear reader, why is Yonatan doing this ugly thing? 
    # This categorization and numbering is a required format for submitting 
    # promotion materials
    pub_count = number_pub(latex_papers["Journal"])
    pub_count = number_pub(latex_papers["Conference"], pub_count)
    pub_count = number_pub(latex_papers["Workshop"], pub_count)
    number_pub(latex_papers["Preprint"], pub_count)

    if args.onepager:
      latex_render = latex_template.render(pub_types=["Conference"], 
                                           publications=latex_papers,
                                           students=student_yaml,
                                           teaching=teach_yaml,
                                           areachair=areachair_yaml,
                                           theses=theses_yaml)
    else:
      latex_render = latex_template.render(pub_types=["Journal", "Conference", "Workshop", "Preprint"], 
                                           publications=latex_papers,
                                           students=student_yaml,
                                           teaching=teach_yaml,
                                           areachair=areachair_yaml,
                                           theses=theses_yaml)
    with open("CV.tex", 'wt') as output_file:
        output_file.write(latex_render)
    os.system("pdflatex CV.tex")
    os.system("rm CV.tex CV.log CV.aux CV.out")
    os.system("mv CV.pdf ../")

def generate_group_page():
    ## Generate group page
    with open("templates/group_template.jinja2", 'r') as file:
      group_template = Template(file.read())
    for stud in student_yaml:
        stud["RESEARCH"] = []
        for pub in pubs:
            if stud["NAME"] in pub["AUTHORS"] and pub["TYPE"] != "workshop":
                stud["RESEARCH"].append(pub)
    masters = yaml.load(open("yaml/students/ms_intern.yaml"), Loader=yaml.CLoader)
    alumni = yaml.load(open("yaml/students/alumni.yaml"), Loader=yaml.CLoader)
    alumni_phd = yaml.load(open("yaml/students/phd_alumni.yaml"), Loader=yaml.CLoader)
    group_render = group_template.render(students=student_yaml, alumni=alumni, alumni_phd=alumni_phd, masters=masters)
    with open("../CLAW/index.html", 'wt') as output_file:
        output_file.write(group_render)

def generate_COA():

    ## Create a list of collaborators from the last 48 months with affiliations (for NSF)
    collaborators = {}
    for pub in pubs:
        if int(pub["YEAR"]) > 2024 - 4 and "Open X-" not in pub["TITLE"]:
            for a in pub["AUTHORS"]:
              if a != "Yonatan Bisk" and a not in student_names:
                parts = a.rsplit(' ',1)
                last_first = parts[1] + ", "+ parts[0]
                collaborators[last_first] = max(int(pub["YEAR"]), collaborators.get(last_first, 0))

    aff_yaml = yaml.load(open("yaml/affiliations.yaml"), Loader=yaml.CLoader)
    affiliations = {}
    for entry in aff_yaml:
        affiliations[entry["NAME"]] = entry["affiliation"]
    
    new_entries = len(collaborators)

    # Load the COA excel spreadsheet
    # Heavily ChatGPT based code here
    workbook = openpyxl.load_workbook('templates/COA.xlsx')
    sheet = workbook.active
    start_row = 52

    # Create a black border style
    border_style = Border(
        left=Side(style='thin', color='000000'),
        right=Side(style='thin', color='000000'),
        top=Side(style='thin', color='000000'),
        bottom=Side(style='thin', color='000000')
    )

    # Adjust merged cells ranges
    for merge in list(sheet.merged_cells.ranges):
        min_col, min_row, max_col, max_row = range_boundaries(str(merge))
        if min_row >= start_row:
            sheet.merged_cells.remove(merge)
            new_merge = f"{openpyxl.utils.get_column_letter(min_col)}{min_row + new_entries}:{openpyxl.utils.get_column_letter(max_col)}{max_row + new_entries}"
            sheet.merged_cells.add(new_merge)
    
    # Adjust the TableD5 position (this is the Editorial board)
    table_d5 = None
    for tbl in sheet.tables.values():
        if tbl.name == 'TableD5':
            table_d5 = tbl
            break

    if table_d5:
        min_col, min_row, max_col, max_row = range_boundaries(table_d5.ref)
        new_ref = f"{get_column_letter(min_col)}{min_row + new_entries}:{get_column_letter(max_col)}{max_row + new_entries}"
        table_d5.ref = new_ref

    sheet.insert_rows(start_row, amount=new_entries)

    for i, author in enumerate(collaborators, start=start_row):
        data = ['A:', 
                author, 
                affiliations[author] if author in affiliations else '',
                '',
                '1/1/' + str(collaborators[author])
                ]
        for col_num, value in enumerate(data, start=1):
            cell = sheet.cell(row=i, column=col_num, value=value)
            cell.border = border_style
    workbook.save('current_COA.xlsx')

def main():
    if not args.onepager:
        generate_publications_website()
    generate_CV()
    if not args.onepager:
        generate_group_page()
        generate_COA()
    generate_shorts()

if __name__ == "__main__":
    main()
